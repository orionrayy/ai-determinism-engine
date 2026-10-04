#!/usr/bin/env python3
"""Credential-optional public research provider fabric."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping

try:
    from .evidence_records import (
        count_independent_sources,
        deduplicate_sources,
        normalize_source,
    )
except ImportError:
    from evidence_records import (
        count_independent_sources,
        deduplicate_sources,
        normalize_source,
    )

MAX_RESPONSE_BYTES = 512 * 1024
DEFAULT_MAX_RESULTS = 8
DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
MAX_CACHE_ENTRY_BYTES = 768 * 1024
MAX_CACHE_FILES = 128

PROVIDER_CONCURRENCY = {
    "crossref": 1,
    "semantic_scholar": 2,
    "europe_pmc": 2,
    "openalex": 2,
    "core": 2,
}
_PROVIDER_SEMAPHORES = {
    provider: threading.BoundedSemaphore(limit)
    for provider, limit in PROVIDER_CONCURRENCY.items()
}

PROVIDER_ORDER = (
    "openalex",
    "semantic_scholar",
    "europe_pmc",
    "crossref",
    "core",
)
FREE_PROVIDER_ORDER = (
    "semantic_scholar",
    "europe_pmc",
    "crossref",
    "openalex",
)
METERED_FREE_PROVIDER_ORDER = (
    "openalex",
)


def default_provider_order() -> tuple[str, ...]:
    free_only = os.environ.get(
        "ORCHESTRATOR_FREE_ONLY", "true"
    ).strip().lower() == "true"
    return FREE_PROVIDER_ORDER if free_only else PROVIDER_ORDER


class ResearchProviderError(RuntimeError):
    pass


def _cache_enabled() -> bool:
    return os.environ.get("ORCHESTRATOR_RESEARCH_CACHE", "true").strip().lower() != "false"


def _cache_dir() -> Path:
    configured = os.environ.get("ORCHESTRATOR_RESEARCH_CACHE_DIR", "").strip()
    return Path(configured) if configured else Path(".orchestrator") / "research-cache"


def _cache_ttl_seconds() -> int:
    raw = os.environ.get("ORCHESTRATOR_RESEARCH_CACHE_TTL", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_CACHE_TTL_SECONDS
    except ValueError:
        value = DEFAULT_CACHE_TTL_SECONDS
    return max(0, min(value, 7 * 24 * 60 * 60))


def _cache_path(provider: str, query: str, max_results: int) -> Path:
    identity = json.dumps(
        {
            "schema": 1,
            "provider": str(provider).strip().lower(),
            "query": str(query).strip(),
            "max_results": int(max_results),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _cache_dir() / (hashlib.sha256(identity).hexdigest() + ".json")


def _cache_load(provider: str, query: str, max_results: int) -> dict[str, Any] | None:
    if not _cache_enabled():
        return None
    path = _cache_path(provider, query, max_results)
    try:
        if time.time() - path.stat().st_mtime > _cache_ttl_seconds():
            return None
        raw = path.read_bytes()
        if len(raw) > MAX_CACHE_ENTRY_BYTES:
            return None
        value = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema") != 1:
        return None
    try:
        cached_max_results = int(value.get("max_results") or 0)
    except (TypeError, ValueError):
        return None
    if (
        str(value.get("provider") or "") != str(provider).strip().lower()
        or str(value.get("query") or "") != str(query).strip()
        or cached_max_results != int(max_results)
    ):
        return None
    payload = value.get("payload")
    return payload if isinstance(payload, dict) else None


def _cache_store(provider: str, query: str, max_results: int, payload: dict[str, Any]) -> None:
    if not _cache_enabled():
        return
    try:
        directory = _cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = _cache_path(provider, query, max_results)
        value = {
            "schema": 1,
            "provider": str(provider).strip().lower(),
            "query": str(query).strip(),
            "max_results": int(max_results),
            "payload": payload,
        }
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_CACHE_ENTRY_BYTES:
            return
        fd, tmp_name = tempfile.mkstemp(prefix=".research-cache-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
        files = sorted(
            (item for item in directory.glob("*.json") if item.is_file()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for stale in files[MAX_CACHE_FILES:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        # Research cache is an optimization only; provider failures must retain
        # the normal execution path rather than turning into cache failures.
        return


def _request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ai-orchestrator-research/2.0",
            **(headers or {}),
        },
        method="GET",
    )
    max_transient_retries = 2
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ResearchProviderError(
                        "research provider response exceeds 512 KiB"
                    )
                if not raw:
                    return {}
                value = json.loads(raw.decode("utf-8", "replace"))
                if not isinstance(value, dict):
                    raise ResearchProviderError(
                        "research provider returned a non-object"
                    )
                return value
        except ResearchProviderError:
            raise
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt >= max_transient_retries:
                raise ResearchProviderError(
                    f"research provider HTTP {exc.code}: {exc.reason}"
                ) from exc
            retry_after = str(exc.headers.get("Retry-After") or "").strip()
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                delay = float(2 ** attempt)
            delay = max(0.0, min(delay, 8.0))
            if delay:
                time.sleep(delay)
            attempt += 1
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= max_transient_retries:
                raise ResearchProviderError(
                    "research provider transient request failure"
                ) from exc
            time.sleep(float(2 ** attempt))
            attempt += 1
        except Exception as exc:
            raise ResearchProviderError(
                f"research provider request failed: {exc}"
            ) from exc


def search_crossref(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    params = {
        "query": query[:400],
        "rows": max(1, min(int(max_results), 50)),
    }
    mailto = os.environ.get("CROSSREF_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto
    return _request_json(
        "https://api.crossref.org/works?" + urllib.parse.urlencode(params),
        headers=(
            {"User-Agent": "ai-orchestrator-research/2.0; mailto=" + mailto}
            if mailto
            else None
        ),
    )


def search_openalex(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    params = {
        "search": query[:400],
        "per-page": str(max(1, min(int(max_results), 50))),
        "select": "id,doi,title,authorships,publication_year,primary_location,cited_by_count,open_access,best_oa_location",
    }
    # In hard free-only mode, deliberately omit a user-supplied OpenAlex key so
    # the engine cannot silently consume prepaid/paid balance. Anonymous usage
    # remains zero-dollar but is subject to OpenAlex's stricter public budget.
    free_only = os.environ.get(
        "ORCHESTRATOR_FREE_ONLY", "true"
    ).strip().lower() == "true"
    key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if key and not free_only:
        params["api_key"] = key
    mailto = os.environ.get("OPENALEX_MAILTO", "").strip()
    if mailto:
        params["mailto"] = mailto[:256]
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    return _request_json(url)


def search_semantic_scholar(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    params = urllib.parse.urlencode({
        "query": query[:400],
        "limit": max(1, min(int(max_results), 100)),
        "fields": "paperId,title,year,authors,citationCount,openAccessPdf,url,journal,abstract",
    })
    headers = {}
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        headers["x-api-key"] = key
    return _request_json(
        "https://api.semanticscholar.org/graph/v1/paper/search?" + params,
        headers=headers,
    )


def search_europe_pmc(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    params = urllib.parse.urlencode({
        "query": query[:400],
        "format": "json",
        "pageSize": max(1, min(int(max_results), 100)),
        "resultType": "core",
    })
    return _request_json(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + params,
    )


def search_core(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    key = os.environ.get("CORE_API_KEY", "").strip()
    if not key:
        raise ResearchProviderError("CORE_API_KEY is not configured")
    params = urllib.parse.urlencode({
        "q": query[:400],
        "limit": max(1, min(int(max_results), 100)),
    })
    return _request_json(
        "https://api.core.ac.uk/v3/search/works?" + params,
        headers={"Authorization": "Bearer " + key},
    )


def _core_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    results = payload.get("results")
    if isinstance(results, list):
        return [item for item in results if isinstance(item, Mapping)]
    return []


def normalize_provider_payload(
    provider: str,
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    provider = str(provider).strip().lower()
    raw_items: list[Mapping[str, Any]] = []
    if provider == "openalex":
        results = payload.get("results")
        if isinstance(results, list):
            raw_items = [item for item in results if isinstance(item, Mapping)]
    elif provider == "semantic_scholar":
        results = payload.get("data")
        if isinstance(results, list):
            raw_items = [item for item in results if isinstance(item, Mapping)]
    elif provider == "crossref":
        message = payload.get("message")
        results = message.get("items") if isinstance(message, Mapping) else None
        if isinstance(results, list):
            raw_items = [item for item in results if isinstance(item, Mapping)]
    elif provider == "europe_pmc":
        result_list = payload.get("resultList", {})
        results = result_list.get("result") if isinstance(result_list, Mapping) else None
        if isinstance(results, list):
            raw_items = [item for item in results if isinstance(item, Mapping)]
    elif provider == "core":
        raw_items = _core_records(payload)
    else:
        raise ResearchProviderError(f"unsupported research provider: {provider}")

    records = []
    for item in raw_items:
        record = dict(item)
        if provider == "core":
            record["provider"] = "core"
            if record.get("id") is not None:
                record["core_id"] = str(record["id"])
        records.append(normalize_source(provider, record))
    return records


def _provider_search(provider: str, query: str, max_results: int) -> dict[str, Any]:
    provider = str(provider).strip().lower()
    cached = _cache_load(provider, query, max_results)
    if cached is not None:
        return cached
    functions = {
        "openalex": search_openalex,
        "semantic_scholar": search_semantic_scholar,
        "europe_pmc": search_europe_pmc,
        "crossref": search_crossref,
        "core": search_core,
    }
    fn = functions.get(provider)
    if fn is None:
        raise ResearchProviderError(f"unsupported research provider: {provider}")
    semaphore = _PROVIDER_SEMAPHORES.get(provider)
    if semaphore is None:
        payload = fn(query, max_results=max_results)
    else:
        with semaphore:
            payload = fn(query, max_results=max_results)
    if isinstance(payload, dict):
        _cache_store(provider, query, max_results, payload)
    return payload


def research_records(
    query: str,
    *,
    providers: tuple[str, ...] | None = None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        raise ValueError("research query is required")
    requested_providers = (
        default_provider_order()
        if providers is None
        else providers
    )
    free_only = os.environ.get(
        "ORCHESTRATOR_FREE_ONLY", "true"
    ).strip().lower() == "true"
    allowed = set(FREE_PROVIDER_ORDER if free_only else PROVIDER_ORDER)
    normalized_providers: list[str] = []
    seen_providers: set[str] = set()
    for item in requested_providers:
        provider = str(item).strip().lower()
        if provider not in allowed:
            continue
        if provider and provider not in seen_providers:
            normalized_providers.append(provider)
            seen_providers.add(provider)
    if providers is not None and not normalized_providers:
        raise ResearchProviderError(
            "no requested research providers are allowed by the current free-only policy"
        )
    records: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    provider_counts: dict[str, int] = {}
    results: dict[str, dict[str, Any]] = {}
    max_workers = max(1, min(4, len(normalized_providers)))
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="research-provider",
    ) as pool:
        futures = {
            pool.submit(
                _provider_search,
                provider,
                query,
                max_results,
            ): provider
            for provider in normalized_providers
        }
        for future in as_completed(futures):
            provider = futures[future]
            try:
                results[provider] = future.result()
            except Exception as exc:
                errors[provider] = str(exc)

    for provider in normalized_providers:
        payload = results.get(provider)
        if payload is None:
            continue
        try:
            normalized = normalize_provider_payload(provider, payload)
            records.extend(normalized)
            provider_counts[provider] = len(normalized)
        except Exception as exc:
            errors[provider] = str(exc)

    deduped = deduplicate_sources(records)
    return {
        "query": query,
        "providers": normalized_providers,
        "evidence_records": deduped,
        "independent_source_count": count_independent_sources(deduped),
        "provider_counts": provider_counts,
        "errors": errors,
    }


__all__ = [
    "ResearchProviderError",
    "search_crossref",
    "search_openalex",
    "search_semantic_scholar",
    "search_europe_pmc",
    "search_core",
    "normalize_provider_payload",
    "research_records",
    "default_provider_order",
    "PROVIDER_CONCURRENCY",
    "_cache_load",
    "_cache_store",
]