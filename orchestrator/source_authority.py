#!/usr/bin/env python3
"""Deterministic source-authority scoring for evidence records.

The score is an explicit, explainable heuristic. It is not a truth oracle:
authority describes the evidentiary standing of a source class, while claim
support and independence remain separate checks.
"""
from __future__ import annotations

from typing import Any, Mapping


AUTHORITY_CLASSES = {
    "official_primary",
    "peer_reviewed",
    "scholarly_article",
    "preprint",
    "institutional",
    "index",
    "community",
    "unknown",
}


def _class_from_record(provider: str, record: Mapping[str, Any]) -> str:
    explicit = str(record.get("authority_class") or "").strip().lower()
    if explicit in AUTHORITY_CLASSES:
        return explicit

    if record.get("peer_reviewed") is True:
        return "peer_reviewed"

    source_type = str(
        record.get("source_type") or record.get("type") or ""
    ).strip().lower()
    if source_type in {"journal-article", "journal article", "article"}:
        return "scholarly_article"

    provider = str(provider or "").strip().lower()
    if provider == "arxiv" or "preprint_repository" in set(
        record.get("authority_signals") or []
    ):
        return "preprint"
    if provider == "europe_pmc" or "biomedical_index" in set(
        record.get("authority_signals") or []
    ):
        return "institutional"
    if provider in {"openalex", "semantic_scholar", "crossref"}:
        return "index"
    if provider == "core":
        return "index"
    return "unknown"


def source_authority(
    provider: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    authority_class = _class_from_record(provider, record)
    score = {
        "official_primary": 0.98,
        "peer_reviewed": 0.90,
        "scholarly_article": 0.78,
        "institutional": 0.76,
        "preprint": 0.58,
        "index": 0.52,
        "community": 0.35,
        "unknown": 0.20,
    }[authority_class]

    signals = {
        str(item).strip().lower()
        for item in (record.get("authority_signals") or [])
        if str(item).strip()
    }
    if "peer_reviewed" in signals:
        score = max(score, 0.90)
    if "journal_metadata" in signals:
        score = max(score, 0.78)

    if score >= 0.85:
        tier = "tier1"
    elif score >= 0.65:
        tier = "tier2"
    elif score >= 0.45:
        tier = "tier3"
    else:
        tier = "tier4"

    return {
        "authority_class": authority_class,
        "authority_score": round(float(score), 3),
        "authority_tier": tier,
        "authority_heuristic": True,
    }


__all__ = ["AUTHORITY_CLASSES", "source_authority"]
