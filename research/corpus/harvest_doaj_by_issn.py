#!/usr/bin/env python3
"""Discover bounded DOAJ article metadata by exact journal ISSN, without fetching content."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any
from urllib.parse import quote, urlsplit

from harvest_doaj_candidates import normalize_issn

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT = 15.0
DEFAULT_MIN_INTERVAL = 1.0
DEFAULT_PER_ISSN = 10


def _registry_sources(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        entries = data.get("sources", data.get("entries", []))
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("source_registry_must_be_object_or_array")
    targets: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("source_type") != "open_access_journal_directory":
            continue
        source_url = str(entry.get("source_url", ""))
        issn_text = urlsplit(source_url).path.rstrip("/").split("/")[-1]
        issn = normalize_issn(issn_text)
        if not issn or issn in seen:
            continue
        seen.add(issn)
        targets.append({
            "issn": issn,
            "source_id": str(entry.get("source_id", "")),
            "journal_name": str(entry.get("name", "")),
            "source_url": source_url,
        })
    return targets


def _get_response(opener: Callable[..., Any], url: str, timeout: float) -> dict:
    request = __import__("urllib.request", fromlist=["Request"]).Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Hallmark-Open-Metadata-Discovery/1.0",
        },
    )
    with opener(request, timeout=timeout) as response:
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if not raw:
        raise ValueError("empty_api_response")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("api_response_exceeded_8MiB_limit")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_json_response") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results", []), list):
        raise ValueError("unexpected_doaj_search_response_shape")
    return payload


def _doi_from_bibjson(bibjson: dict) -> str | None:
    identifiers = bibjson.get("identifier") or []
    if not isinstance(identifiers, list):
        return None
    for item in identifiers:
        if isinstance(item, dict) and str(item.get("type", "")).lower() == "doi":
            value = item.get("id")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _candidate_from_result(result: dict, target: dict[str, str], queried_issn: str) -> dict | None:
    if not isinstance(result, dict):
        return None
    article_id = result.get("id")
    bibjson = result.get("bibjson")
    if not isinstance(article_id, str) or not article_id.strip() or not isinstance(bibjson, dict):
        return None
    title = bibjson.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    journal = bibjson.get("journal") if isinstance(bibjson.get("journal"), dict) else {}
    journal_issns = journal.get("issns") if isinstance(journal.get("issns"), list) else []
    clean_issns = sorted({
        norm for value in journal_issns
        if isinstance(value, str) and (norm := normalize_issn(value))
    })
    creators = bibjson.get("author") if isinstance(bibjson.get("author"), list) else []
    clean_creators = []
    for creator in creators:
        if isinstance(creator, dict):
            name = creator.get("name")
            if isinstance(name, str) and name.strip():
                clean_creators.append(name.strip())
    links = bibjson.get("link") if isinstance(bibjson.get("link"), list) else []
    full_text_urls = sorted({
        item["url"].strip()
        for item in links
        if isinstance(item, dict)
        and isinstance(item.get("url"), str)
        and item["url"].strip().startswith(("https://", "http://"))
        and str(item.get("type", "")).lower() == "fulltext"
    })
    article_url = f"https://doaj.org/article/{article_id.strip()}"
    return {
        "candidate_id": "doaj-article-" + article_id.strip(),
        "doaj_article_id": article_id.strip(),
        "article_url": article_url,
        "title": title.strip(),
        "creators": clean_creators,
        "journal_title": str(journal.get("title") or target["journal_name"]),
        "journal_publisher": str(journal.get("publisher") or ""),
        "publication_year": str(bibjson.get("year") or ""),
        "doi": _doi_from_bibjson(bibjson),
        "journal_issns": clean_issns,
        "matched_target_issns": [queried_issn],
        "full_text_urls_metadata_only": full_text_urls,
        "discovery_source": "DOAJ public Search API by journal ISSN",
        "target_journal_registry_id": target["source_id"],
        "article_license_status": "not_verified_for_this_article",
        "corpus_eligible": False,
        "abstract_saved": False,
        "full_text_downloaded": False,
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as tmp:
        tmp.write(content)
        temporary = Path(tmp.name)
    os.replace(temporary, path)


def discover(
    registry_path: Path,
    *,
    per_issn: int = DEFAULT_PER_ISSN,
    timeout: float = DEFAULT_TIMEOUT,
    min_interval: float = DEFAULT_MIN_INTERVAL,
    opener: Callable[..., Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict:
    if per_issn < 1 or per_issn > 50:
        raise ValueError("per_issn_must_be_between_1_and_50")
    if timeout <= 0 or timeout > 30:
        raise ValueError("timeout_must_be_between_0_and_30_seconds")
    if min_interval < 0.5:
        raise ValueError("min_interval_must_be_at_least_0.5_seconds_to_respect_DOAJ_rate_limit")
    if opener is None:
        from urllib.request import urlopen
        opener = urlopen

    targets = _registry_sources(registry_path)
    if not targets:
        raise ValueError("no_valid_DOAJ_ISSN_targets_in_source_registry")

    by_id: dict[str, dict] = {}
    source_results = []
    errors = []
    request_count = 0
    total_reported = 0
    last_request_at: float | None = None

    for target in targets:
        issn = target["issn"]
        if last_request_at is not None:
            elapsed = time.monotonic() - last_request_at
            if elapsed < min_interval:
                sleeper(min_interval - elapsed)
        url = f"https://doaj.org/api/search/articles/issn:{quote(issn, safe='-Xx')}?page=1&pageSize={per_issn}"
        try:
            last_request_at = time.monotonic()
            payload = _get_response(opener, url, timeout)
            request_count += 1
            total = payload.get("total", 0)
            if not isinstance(total, int) or total < 0:
                total = 0
            total_reported += total
            results = payload.get("results", [])
            added_this_source = 0
            for item in results:
                candidate = _candidate_from_result(item, target, issn)
                if candidate is None:
                    continue
                candidate_id = candidate["candidate_id"]
                if candidate_id not in by_id:
                    by_id[candidate_id] = candidate
                    added_this_source += 1
                else:
                    current = by_id[candidate_id]
                    current["matched_target_issns"] = sorted(set(current["matched_target_issns"]) | {issn})
            source_results.append({
                "issn": issn,
                "source_id": target["source_id"],
                "journal_name": target["journal_name"],
                "query_succeeded": True,
                "api_total_matches": total,
                "api_records_returned": len(results),
                "unique_candidates_added": added_this_source,
            })
        except Exception as exc:  # per-target failure must not erase other successful metadata
            detail = f"{type(exc).__name__}: {str(exc)}"[:240]
            errors.append({"issn": issn, "source_id": target["source_id"], "error": detail})
            source_results.append({
                "issn": issn,
                "source_id": target["source_id"],
                "journal_name": target["journal_name"],
                "query_succeeded": False,
                "api_total_matches": None,
                "api_records_returned": 0,
                "unique_candidates_added": 0,
                "error": detail,
            })

    candidate_rows = [by_id[key] for key in sorted(by_id)]
    successes = sum(1 for source in source_results if source["query_succeeded"])
    if successes == 0:
        status = "LIVE_DISCOVERY_UNAVAILABLE"
    elif errors:
        status = "PARTIAL"
    elif not candidate_rows:
        status = "NO_NEW_CANDIDATES"
    else:
        status = "CANDIDATES_DISCOVERED"

    summary = {
        "status": status,
        "source": "DOAJ public Search API",
        "query_method": "one exact ISSN query per journal registry entry",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "target_journal_count": len(targets),
        "target_issns": [target["issn"] for target in targets],
        "successful_queries": successes,
        "failed_queries": len(errors),
        "api_records_reported_across_queries": total_reported,
        "unique_candidate_records_written": len(candidate_rows),
        "per_issn_requested": per_issn,
        "request_count": request_count + len(errors),
        "request_min_interval_seconds": min_interval,
        "article_content_downloaded": False,
        "abstracts_retained": False,
        "candidate_inventory_is_corpus": False,
        "content_license_warning": "Journal-level DOAJ license metadata does not verify an individual article's license. Review the exact article and rights statement before promoting any candidate.",
        "target_500_status": "INSUFFICIENT_EVIDENCE",
        "sources": source_results,
        "errors": errors,
    }
    return {"summary": summary, "candidates": candidate_rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path(__file__).with_name("source_registry.json"))
    parser.add_argument("--per-issn", type=int, default=DEFAULT_PER_ISSN)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL)
    parser.add_argument("--output", type=Path, default=Path("candidates-by-issn.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("doaj-by-issn-summary.json"))
    args = parser.parse_args(argv)
    try:
        result = discover(
            args.registry, per_issn=args.per_issn, timeout=args.timeout,
            min_interval=args.min_interval,
        )
    except Exception as exc:
        print(f"doaj_by_issn_discovery_failed: {type(exc).__name__}: {str(exc)[:240]}", file=sys.stderr)
        return 2

    jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in result["candidates"])
    _atomic_write(args.output, jsonl)
    _atomic_write(args.summary, json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["summary"]["successful_queries"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
