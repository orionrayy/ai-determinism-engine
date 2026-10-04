#!/usr/bin/env python3
"""Canonical, deterministic evidence records for free-first research."""
# v67 verification: provider-neutral canonical identity boundary.
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping
try:
    from .evidence_independence import source_work_identity
except ImportError:
    from evidence_independence import source_work_identity


_DOI_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", re.IGNORECASE)
_ARXIV_RE = re.compile(r"(?:https?://arxiv\.org/(?:abs|pdf)/)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)$", re.IGNORECASE)


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _norm_doi(value: Any) -> str:
    raw = _norm_text(value)
    if not raw:
        return ""
    raw = _DOI_RE.sub("", raw).strip().rstrip(".")
    return raw


def _norm_arxiv(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    match = _ARXIV_RE.search(raw)
    return match.group(1).lower() if match else raw.lower()


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _authors(record: Mapping[str, Any]) -> list[str]:
    raw = record.get("authors")
    if isinstance(raw, list):
        names = []
        for item in raw:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, Mapping):
                name = _first_nonempty(
                    item.get("name"),
                    item.get("display_name"),
                    item.get("family"),
                )
                name = str(name or "").strip()
            else:
                name = ""
            if name:
                names.append(name)
        return names
    raw = record.get("authorships")
    if isinstance(raw, list):
        names = []
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            author = item.get("author")
            if isinstance(author, Mapping):
                name = _first_nonempty(author.get("display_name"), author.get("name"))
                if name:
                    names.append(str(name).strip())
        return names
    raw = record.get("author")
    if isinstance(raw, list):
        names = []
        for item in raw:
            if isinstance(item, Mapping):
                name = _first_nonempty(item.get("name"), item.get("family"), item.get("display_name"))
                if name:
                    names.append(str(name).strip())
        return names
    return []


def _published(record: Mapping[str, Any]) -> str:
    raw = _first_nonempty(
        record.get("published"),
        record.get("published_online"),
        record.get("publication_date"),
    )
    if isinstance(raw, Mapping):
        parts = raw.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            return "-".join(str(x) for x in parts[0])
    if isinstance(raw, str):
        return raw.strip()
    year = _first_nonempty(record.get("year"), record.get("publication_year"))
    return str(year) if year is not None else ""


def _venue(record: Mapping[str, Any]) -> str:
    primary = record.get("primary_location")
    if isinstance(primary, Mapping):
        source = primary.get("source")
        if isinstance(source, Mapping):
            name = _first_nonempty(source.get("display_name"), source.get("name"))
            if name:
                return str(name)
    journal = record.get("journal")
    if isinstance(journal, Mapping):
        name = _first_nonempty(journal.get("name"), journal.get("display_name"))
        if name:
            return str(name)
    return str(_first_nonempty(
        record.get("container-title"),
        record.get("venue"),
    ) or "")


def _full_text_url(record: Mapping[str, Any]) -> str:
    oa_pdf = record.get("openAccessPdf")
    if isinstance(oa_pdf, Mapping):
        value = _first_nonempty(oa_pdf.get("url"))
        if value:
            return str(value)
    best_oa = record.get("best_oa_location")
    if isinstance(best_oa, Mapping):
        for key in ("pdf_url", "landing_page_url", "url"):
            value = best_oa.get(key)
            if value:
                return str(value)
    for key in ("full_text_url", "pdf_url"):
        value = record.get(key)
        if value:
            return str(value)
    links = record.get("link")
    if isinstance(links, list):
        for item in links:
            if not isinstance(item, Mapping):
                continue
            value = item.get("URL") or item.get("url")
            content_type = str(
                item.get("content-type") or item.get("content_type") or ""
            ).lower()
            if value and (
                "pdf" in content_type
                or "fulltext" in content_type
                or "xhtml" in content_type
            ):
                return str(value)
    return ""


def _abstract_text(record: Mapping[str, Any]) -> str:
    value = _first_nonempty(
        record.get("abstract"),
        record.get("abstractText"),
    )
    if not value:
        return ""
    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())[:16000]


def _access_metadata(
    provider: str,
    record: Mapping[str, Any],
    abstract: str,
    full_text_url: str,
) -> dict[str, Any]:
    explicit_full_text = bool(full_text_url)
    best_oa = record.get("best_oa_location")
    has_oa_pdf = (
        isinstance(record.get("openAccessPdf"), Mapping)
        and bool(record.get("openAccessPdf", {}).get("url"))
    ) or (
        isinstance(best_oa, Mapping)
        and bool(best_oa.get("pdf_url"))
    )
    publisher_url = str(record.get("URL") or record.get("url") or "").strip()

    if has_oa_pdf:
        access_level = "L3"
        access_route = "open_access_pdf"
    elif explicit_full_text:
        access_level = "L3"
        access_route = "full_text_url"
    elif abstract:
        access_level = "L2"
        access_route = "abstract"
    elif publisher_url:
        access_level = "L1"
        access_route = "publisher_url"
    elif full_text_url:
        access_level = "L1"
        access_route = "locator"
    else:
        access_level = "L0"
        access_route = "identifier"

    source_type = _norm_text(
        _first_nonempty(record.get("source_type"), record.get("type"))
    )
    peer_reviewed = record.get("peer_reviewed") is True
    if peer_reviewed:
        authority_tier = "peer_review_signal"
        authority_score = 0.80
    elif provider == "arxiv":
        authority_tier = "preprint_repository"
        authority_score = 0.45
    elif provider == "europe_pmc":
        authority_tier = "biomedical_index"
        authority_score = 0.55
    elif provider in {"crossref", "openalex", "semantic_scholar"}:
        authority_tier = "scholarly_index"
        authority_score = 0.55
    elif source_type in {"journal-article", "journal article"}:
        authority_tier = "journal_metadata"
        authority_score = 0.60
    else:
        authority_tier = "metadata_index"
        authority_score = 0.40

    return {
        "access_level": access_level,
        "access_route": access_route,
        "authority_tier": authority_tier,
        "authority_score": authority_score,
        "authority_heuristic": True,
    }


def _identifiers(record: Mapping[str, Any]) -> dict[str, str]:
    doi = _norm_doi(_first_nonempty(record.get("DOI"), record.get("doi")))
    arxiv_id = _norm_arxiv(_first_nonempty(record.get("arxiv_id"), record.get("arxivId"), record.get("arxiv")))
    pmid = str(_first_nonempty(record.get("pmid"), record.get("PMID")) or "").strip().lower()
    pmcid = str(_first_nonempty(record.get("pmcid"), record.get("PMCID")) or "").strip().lower()
    core_id = str(_first_nonempty(record.get("core_id"), record.get("id") if record.get("provider") == "core" else None) or "").strip()
    return {
        "doi": doi,
        "arxiv_id": arxiv_id,
        "pmid": pmid,
        "pmcid": pmcid,
        "core_id": core_id,
    }


def canonical_source_id(record: Mapping[str, Any]) -> str:
    existing = _norm_text(record.get("canonical_id"))
    if existing:
        return existing
    ids = _identifiers(record)
    if ids["doi"]:
        return "doi:" + ids["doi"]
    if ids["arxiv_id"]:
        return "arxiv:" + ids["arxiv_id"]
    if ids["pmid"]:
        return "pmid:" + ids["pmid"]
    if ids["pmcid"]:
        return "pmcid:" + ids["pmcid"]
    if ids["core_id"]:
        return "core:" + ids["core_id"]

    title = _norm_text(record.get("title"))
    authors = _authors(record)
    year = str(_first_nonempty(record.get("year"), record.get("publication_year"), "") or "")
    if title and (authors or year):
        anchor = {
            "title": title,
            "authors": [_norm_text(name) for name in authors[:8]],
            "year": year,
        }
        digest = hashlib.sha256(
            ("bibliographic:" + repr(sorted(anchor.items()))).encode("utf-8")
        ).hexdigest()[:32]
        return "fingerprint:" + digest

    provider = _norm_text(record.get("provider"))
    provider_id = _norm_text(record.get("provider_id") or record.get("paperId"))
    if provider_id:
        return f"{provider}:{provider_id}"
    digest = hashlib.sha256(("fallback:" + title).encode("utf-8")).hexdigest()[:32]
    return "fingerprint:" + digest


def normalize_source(provider: str, record: Mapping[str, Any]) -> dict[str, Any]:
    provider = _norm_text(provider).replace(" ", "_")
    ids = _identifiers(record)
    title = str(_first_nonempty(record.get("title"), record.get("display_name")) or "").strip()
    authors = _authors(record)
    citations_raw = _first_nonempty(record.get("citationCount"), record.get("cited_by_count"), record.get("citation_count"))
    try:
        citation_count = max(0, int(citations_raw)) if citations_raw is not None else 0
    except (TypeError, ValueError):
        citation_count = 0
    oa_raw = record.get("open_access")
    open_access = bool(
        oa_raw is True
        or (isinstance(oa_raw, Mapping) and oa_raw.get("is_oa") is True)
        or record.get("openAccessPdf")
        or record.get("full_text_url")
        or record.get("pdf_url")
    )
    year = _first_nonempty(record.get("year"), record.get("publication_year"))
    source_type = _norm_text(_first_nonempty(record.get("source_type"), record.get("type")))
    primaryity = str(_first_nonempty(record.get("primaryity"), "unknown"))
    abstract = _abstract_text(record)
    authority = []
    if source_type in {"journal-article", "journal article"}:
        authority.append("journal_metadata")
    if provider in {"crossref", "openalex", "semantic_scholar"}:
        authority.append("scholarly_index")
    if provider == "arxiv":
        authority.append("preprint_repository")
    if provider == "europe_pmc":
        authority.append("biomedical_index")
    if provider == "core":
        authority.append("open_access_index")
    if record.get("peer_reviewed") is True:
        authority.append("peer_reviewed")
    work_identity = source_work_identity({**record, "provider": provider})
    access = _access_metadata(
        provider,
        record,
        abstract,
        _full_text_url(record),
    )
    return {
        "canonical_id": canonical_source_id({**record, "provider": provider}),
        "provider": provider,
        "provider_id": str(_first_nonempty(record.get("provider_id"), record.get("paperId"), record.get("id")) or ""),
        "providers": [provider],
        "title": title,
        "authors": authors,
        "published": _published(record),
        "year": int(year) if str(year or "").isdigit() else None,
        **ids,
        "venue": _venue(record),
        "abstract": abstract,
        **access,
        "citation_count": citation_count,
        "open_access": open_access,
        "full_text_url": _full_text_url(record),
        "primaryity": primaryity,
        "authority_signals": sorted(set(authority)),
        "independence_key": str(work_identity["work_key"]),
        "independence_basis": str(work_identity["basis"]),
        "independence_confidence": float(work_identity["confidence"]),
        "independence_proxy": True,
    }


def deduplicate_sources(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        cid = canonical_source_id(record)
        current = merged.get(cid)
        if current is None:
            current = dict(record)
            current["canonical_id"] = cid
            current["providers"] = sorted(set(record.get("providers") or [record.get("provider", "")]))
            merged[cid] = current
            continue
        current["providers"] = sorted(set(current.get("providers") or []) | set(record.get("providers") or [record.get("provider", "")]))
        current["authority_signals"] = sorted(set(current.get("authority_signals") or []) | set(record.get("authority_signals") or []))
        for key in ("title", "published", "venue", "doi", "arxiv_id", "pmid", "pmcid", "core_id", "full_text_url"):
            if not current.get(key) and record.get(key):
                current[key] = record[key]
        current["authors"] = current.get("authors") or record.get("authors") or []
        current["citation_count"] = max(
            int(current.get("citation_count") or 0),
            int(record.get("citation_count") or 0),
        )
        current["open_access"] = bool(current.get("open_access") or record.get("open_access"))
        if current.get("primaryity") in (None, "", "unknown") and record.get("primaryity"):
            current["primaryity"] = record["primaryity"]
    result = list(merged.values())
    for record in result:
        record["providers"] = sorted(set(record.get("providers") or []))
        record["independence_key"] = str(record.get("independence_key") or record["canonical_id"])
    return sorted(
        result,
        key=lambda item: (str(item.get("canonical_id")), str(item.get("title") or "")),
    )


def count_independent_sources(records: list[Mapping[str, Any]]) -> int:
    """Compatibility alias for distinct evidence-work count.

    It is an independence proxy, not proof that studies share no authors,
    datasets, citations, or other dependency relationships.
    """
    return len({
        str(record.get("independence_key") or record.get("canonical_id") or "")
        for record in records
        if str(record.get("independence_key") or record.get("canonical_id") or "")
    })


__all__ = [
    "canonical_source_id",
    "normalize_source",
    "deduplicate_sources",
    "count_independent_sources",
]
