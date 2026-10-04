from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

_WS_RE = re.compile(r"\s+")


def _norm(value: Any) -> str:
    return _WS_RE.sub(" ", str(value or "").strip().lower())


def _authors(record: Mapping[str, Any]) -> list[str]:
    raw = record.get("authors")
    if isinstance(raw, list):
        result = []
        for item in raw:
            if isinstance(item, str):
                name = item
            elif isinstance(item, Mapping):
                name = item.get("name") or item.get("display_name") or item.get("family")
            else:
                name = ""
            if str(name or "").strip():
                result.append(_norm(name))
        return result

    raw = record.get("authorships")
    if isinstance(raw, list):
        result = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            author = item.get("author")
            if isinstance(author, Mapping):
                name = author.get("display_name") or author.get("name")
                if name:
                    result.append(_norm(name))
        return result
    return []


def _identifier(record: Mapping[str, Any]) -> tuple[str, str]:
    for field, prefix in (
        ("doi", "doi"),
        ("DOI", "doi"),
        ("arxiv_id", "arxiv"),
        ("arxivId", "arxiv"),
        ("pmid", "pmid"),
        ("PMID", "pmid"),
        ("pmcid", "pmcid"),
        ("PMCID", "pmcid"),
    ):
        value = _norm(record.get(field))
        if value:
            if prefix == "arxiv":
                value = re.sub(r"v[0-9]+$", "", value)
            return f"{prefix}:{value}", "strong"

    canonical = _norm(record.get("canonical_id"))
    if canonical and any(
        canonical.startswith(prefix + ":")
        for prefix in ("doi", "arxiv", "pmid", "pmcid")
    ):
        return canonical, "strong"

    if canonical:
        return canonical, "canonical_identifier"

    return "", ""


def source_work_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a work-level identity, deliberately not a proof of independence."""
    identifier, basis = _identifier(record)
    if identifier:
        return {
            "work_key": identifier,
            "basis": basis,
            "confidence": 1.0,
        }

    title = _norm(record.get("title") or record.get("display_name"))
    authors = _authors(record)
    year = _norm(record.get("year") or record.get("publication_year"))
    venue = _norm(
        record.get("venue")
        or record.get("container-title")
        or (
            record.get("journal", {}).get("name")
            if isinstance(record.get("journal"), Mapping)
            else ""
        )
    )

    if title and (authors or year):
        anchor = {
            "title": title,
            "authors": authors[:8],
            "year": year,
            "venue": venue,
        }
        raw = repr(
            sorted(anchor.items(), key=lambda item: item[0])
        ).encode("utf-8")
        return {
            "work_key": "work:" + hashlib.sha256(raw).hexdigest()[:32],
            "basis": "heuristic_bibliographic",
            "confidence": 0.8,
        }

    provider = _norm(record.get("provider"))
    provider_id = _norm(
        record.get("provider_id")
        or record.get("paperId")
        or record.get("id")
    )
    if provider_id:
        return {
            "work_key": f"provider:{provider}:{provider_id}",
            "basis": "provider_identifier",
            "confidence": 0.5,
        }

    title_digest = hashlib.sha256(
        ("title:" + title).encode("utf-8")
    ).hexdigest()[:32]
    return {
        "work_key": "weak:" + title_digest,
        "basis": "weak_title_only",
        "confidence": 0.2,
    }


def _bibliographic_crosswalk_key(record: Mapping[str, Any]) -> str:
    title = _norm(record.get("title") or record.get("display_name"))
    authors = _authors(record)
    year = _norm(record.get("year") or record.get("publication_year"))
    if not title or not authors or not year:
        return ""
    payload = {
        "title": title,
        "authors": authors[:8],
        "year": year,
    }
    raw = repr(sorted(payload.items())).encode("utf-8")
    return "bib:" + hashlib.sha256(raw).hexdigest()[:32]


def independence_summary(
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    identity_rows: list[tuple[dict[str, Any], str, str]] = []
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    crosswalk_anchor: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        identity = source_work_identity(record)
        key = str(identity["work_key"])
        find(key)
        crosswalk = _bibliographic_crosswalk_key(record)
        identity_rows.append((identity, key, crosswalk))
        if crosswalk:
            anchor = crosswalk_anchor.get(crosswalk)
            if anchor is None:
                crosswalk_anchor[crosswalk] = key
            else:
                union(anchor, key)

    groups: dict[str, dict[str, Any]] = {}
    for identity, key, crosswalk in identity_rows:
        root = find(key)
        group = groups.setdefault(
            root,
            {
                "work_key": str(identity["work_key"]),
                "basis": str(identity["basis"]),
                "confidence": float(identity["confidence"]),
                "providers": set(),
                "authors": set(),
                "titles": set(),
                "identity_keys": set(),
                "crosswalked": False,
            },
        )
        group["confidence"] = min(
            float(group["confidence"]),
            float(identity["confidence"]),
        )
        group["identity_keys"].add(key)
        group["crosswalked"] = bool(group["crosswalked"] or crosswalk)
        provider = _norm(next(
            (
                record.get("provider")
                for record in records
                if isinstance(record, Mapping)
                and str(source_work_identity(record)["work_key"]) == key
            ),
            "",
        ))
        if provider:
            group["providers"].add(provider)
        # Recover author/title evidence directly from the matching identity rows.
    # Rebuild author/provider/title aggregates in one deterministic pass.
    groups = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        identity = source_work_identity(record)
        key = str(identity["work_key"])
        root = find(key)
        group = groups.setdefault(
            root,
            {
                "work_key": key,
                "basis": str(identity["basis"]),
                "confidence": float(identity["confidence"]),
                "providers": set(),
                "authors": set(),
                "titles": set(),
                "identity_keys": set(),
                "crosswalked": False,
            },
        )
        group["confidence"] = min(
            float(group["confidence"]),
            float(identity["confidence"]),
        )
        group["identity_keys"].add(key)
        group["crosswalked"] = bool(
            group["crosswalked"] or _bibliographic_crosswalk_key(record)
        )
        provider = _norm(record.get("provider"))
        if provider:
            group["providers"].add(provider)
        for author in _authors(record)[:8]:
            if author:
                group["authors"].add(author)
        title = _norm(record.get("title") or record.get("display_name"))
        if title:
            group["titles"].add(title)
        basis_rank = {
            "weak_title_only": 0,
            "provider_identifier": 1,
            "heuristic_bibliographic": 2,
            "canonical_identifier": 3,
            "strong": 4,
        }
        if basis_rank.get(str(identity["basis"]), 0) > basis_rank.get(str(group["basis"]), 0):
            group["basis"] = str(identity["basis"])

    normalized_groups = [
        {
            "work_key": str(group["work_key"]),
            "basis": str(group["basis"]),
            "confidence": round(float(group["confidence"]), 3),
            "providers": sorted(group["providers"]),
            "author_count": len(group["authors"]),
            "title_count": len(group["titles"]),
            "crosswalked": bool(group["crosswalked"] and len(group["identity_keys"]) > 1),
        }
        for group in groups.values()
    ]
    normalized_groups.sort(key=lambda item: item["work_key"])

    strong = sum(item["basis"] == "strong" for item in normalized_groups)
    canonical = sum(item["basis"] == "canonical_identifier" for item in normalized_groups)
    heuristic = sum(item["basis"] == "heuristic_bibliographic" for item in normalized_groups)
    weak = len(normalized_groups) - strong - canonical - heuristic

    avg_confidence = (
        sum(item["confidence"] for item in normalized_groups) / len(normalized_groups)
        if normalized_groups
        else 0.0
    )

    return {
        "distinct_work_count": len(normalized_groups),
        "strong_work_count": strong,
        "canonical_identifier_work_count": canonical,
        "heuristic_work_count": heuristic,
        "weak_work_count": weak,
        "crosswalked_work_count": sum(
            1 for item in normalized_groups if item["crosswalked"]
        ),
        "independence_proxy_confidence": round(avg_confidence, 3),
        "independence_proxy": True,
        "groups": normalized_groups,
    }

def count_distinct_evidence_works(
    records: list[Mapping[str, Any]],
) -> int:
    return int(independence_summary(records)["distinct_work_count"])


__all__ = [
    "count_distinct_evidence_works",
    "independence_summary",
    "source_work_identity",
]
