#!/usr/bin/env python3
"""Credential-optional public research provider fabric."""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
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

PROVIDER_ORDER = (
    "openalex",
    "semantic_scholar",
    "europe_pmc",
    "core",
)
FREE_PROVIDER_ORDER = (
    "semantic_scholar",
    "europe_pmc",
)


def default_provider_order() -> tuple[str, ...]:
    free_only = os.environ.get(
        "ORCHESTRATOR_FREE_ONLY", "true"
    ).strip().lower() == "true"
    return FREE_PROVIDER_ORDER if free_only else PROVIDER_ORDER


class ResearchProviderError(RuntimeError):
    pass


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
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ResearchProviderError("research provider response exceeds 512 KiB")
            if not raw:
                return {}
            value = json.loads(raw.decode("utf-8", "replace"))
            if not isinstance(value, dict):
                raise ResearchProviderError("research provider returned a non-object")
            return value
    except ResearchProviderError:
        raise
    except Exception as exc:
        raise ResearchProviderError(f"research provider request failed: {exc}") from exc


def search_openalex(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    params = {
        "search": query[:400],
        "per-page": str(max(1, min(int(max_results), 50))),
        "select": "id,doi,title,authorships,publication_year,primary_location,cited_by_count,open_access,best_oa_location",
    }
    key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if key:
        params["api_key"] = key
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    return _request_json(url)


def search_semantic_scholar(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    params = urllib.parse.urlencode({
        "query": query[:400],
        "limit": max(1, min(int(max_results), 100)),
        "fields": "paperId,title,year,authors,citationCount,openAccessPdf,url,journal",
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
    functions = {
        "openalex": search_openalex,
        "semantic_scholar": search_semantic_scholar,
        "europe_pmc": search_europe_pmc,
        "core": search_core,
    }
    fn = functions.get(provider)
    if fn is None:
        raise ResearchProviderError(f"unsupported research provider: {provider}")
    return fn(query, max_results=max_results)


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
    normalized_providers: list[str] = []
    seen_providers: set[str] = set()
    for item in requested_providers:
        provider = str(item).strip().lower()
        if provider and provider not in seen_providers:
            normalized_providers.append(provider)
            seen_providers.add(provider)
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
    "search_openalex",
    "search_semantic_scholar",
    "search_europe_pmc",
    "search_core",
    "normalize_provider_payload",
    "research_records",
    "default_provider_order",
]