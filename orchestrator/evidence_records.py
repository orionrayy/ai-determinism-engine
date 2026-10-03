#!/usr/bin/env python3
"""Canonical, deterministic evidence records for free-first research."""
# v67 verification: provider-neutral canonical identity boundary.
from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

_DOI_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", re.IGNORECASE)
_ARXIV_RE = re.compile(r"(?:https?://arxiv\.org/(?:abs|pdf)/)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)$", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^a-z0-9]+")


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