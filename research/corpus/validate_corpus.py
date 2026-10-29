#!/usr/bin/env python3
"""Validate Hallmark's provenance-gated original-document corpus manifest.

Stdlib-only by design. This tool never downloads source documents and never
assumes that public access grants permission to redistribute copyrighted files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TARGET_DEFAULT = 500
GENRES = (
    "nursing_lp_askep", "pkl_internship", "thesis_dissertation",
    "academic_paper", "lab_practicum", "research_report", "proposal",
    "event_report", "professional_work_report", "operational_admin",
    "excel_workbook", "other",
)
ENUMS = {
    "language": {"id", "en", "other"},
    "format": {"pdf", "docx", "xlsx", "xls", "html", "odt", "txt", "csv", "other"},
    "access_basis": {"open_license", "public_domain", "explicit_permission",
                     "terms_allow_analysis", "metadata_only", "unclear", "restricted"},
    "usage_scope": {"full_text_analysis", "feature_analysis_only", "metadata_only", "excluded"},
    "originality_status": {"original_verified", "appears_original_unverified",
                           "official_template", "duplicate_or_version", "synthetic_or_ai"},
    "provenance_confidence": {"high", "medium", "low"},
    "acquisition_method": {"official_download", "official_web_view", "api",
                           "user_supplied_permission", "library_metadata_only", "other"},
    "access_status": {"accessed", "metadata_only", "inaccessible"},
    "anonymization_status": {"not_applicable", "verified", "pending", "excluded"},
    "privacy_review": {"not_required", "passed", "pending", "failed"},
    "verification_status": {"pending", "verified", "excluded"},
}
REQUIRED = {
    "record_id", "source_family_id", "source_url", "title", "creator", "publisher",
    "genre", "subgenre", "publication_year", "retrieved_at", "language", "format",
    "access_basis", "license", "usage_scope", "originality_status",
    "provenance_confidence", "provenance_evidence", "acquisition_method",
    "access_status", "anonymization_status", "privacy_review",
    "verification_status", "sha256", "byte_size", "duplicate_of", "exclusion_reason",
}
OPTIONAL = {"notes"}
ALLOWED_VERIFIED_ACCESS = {
    "open_license", "public_domain", "explicit_permission", "terms_allow_analysis",
}


def canonical_url(raw: str) -> str | None:
    """Normalize a source URL conservatively while retaining meaningful query values."""
    try:
        p = urlsplit(raw.strip())
        if p.scheme.lower() not in {"http", "https"} or not p.hostname or p.username or p.password:
            return None
        host = p.hostname.lower()
        if p.port and not ((p.scheme.lower() == "https" and p.port == 443)
                           or (p.scheme.lower() == "http" and p.port == 80)):
            host += f":{p.port}"
        path = re.sub(r"/{2,}", "/", p.path or "/")
        if path != "/":
            path = path.rstrip("/")
        query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                 if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
        query.sort()
        return urlunsplit((p.scheme.lower(), host, path, urlencode(query, doseq=True), ""))
    except (ValueError, UnicodeError):
        return None


def validate_record(obj: object, line_no: int) -> list[str]:
    prefix = f"line {line_no}"
    if not isinstance(obj, dict):
        return [f"{prefix}: each JSONL row must be an object"]
    errors: list[str] = []
    missing = sorted(REQUIRED - set(obj))
    extra = sorted(set(obj) - REQUIRED - OPTIONAL)
    if missing:
        errors.append(f"{prefix}: missing required fields: {', '.join(missing)}")
    if extra:
        errors.append(f"{prefix}: unknown fields: {', '.join(extra)}")

    def nonempty(field: str, max_len: int = 5000) -> None:
        value = obj.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{prefix}: {field} must be a non-empty string")
        elif len(value) > max_len:
            errors.append(f"{prefix}: {field} exceeds {max_len} characters")

    for field in ("record_id", "source_family_id", "source_url", "title", "publisher",
                  "subgenre", "license"):
        nonempty(field, 2000 if field == "source_url" else 1000)
    if isinstance(obj.get("record_id"), str) and not re.fullmatch(r"[a-z0-9][a-z0-9._:-]{2,159}", obj["record_id"]):
        errors.append(f"{prefix}: record_id must be a stable lowercase identifier")
    if isinstance(obj.get("source_url"), str) and canonical_url(obj["source_url"]) is None:
        errors.append(f"{prefix}: source_url must be an HTTP(S) URL without embedded credentials")
    if obj.get("creator") is not None and not isinstance(obj.get("creator"), str):
        errors.append(f"{prefix}: creator must be a string or null")
    if obj.get("publication_year") is not None:
        year = obj.get("publication_year")
        if type(year) is not int or not 1800 <= year <= 2200:
            errors.append(f"{prefix}: publication_year must be null or an integer in 1800..2200")
    try:
        if not isinstance(obj.get("retrieved_at"), str):
            raise ValueError
        date.fromisoformat(obj["retrieved_at"])
    except (ValueError, TypeError):
        errors.append(f"{prefix}: retrieved_at must be an ISO date YYYY-MM-DD")
    if obj.get("genre") not in GENRES:
        errors.append(f"{prefix}: unsupported genre")
    for field, allowed in ENUMS.items():
        if obj.get(field) not in allowed:
            errors.append(f"{prefix}: {field} must be one of {', '.join(sorted(allowed))}")
    evidence = obj.get("provenance_evidence")
    if not isinstance(evidence, list) or any(not isinstance(x, str) or not x.strip() for x in evidence):
        errors.append(f"{prefix}: provenance_evidence must be an array of non-empty strings")
    sha = obj.get("sha256")
    if sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", sha)):
        errors.append(f"{prefix}: sha256 must be null or 64 hexadecimal characters")
    size = obj.get("byte_size")
    if size is not None and (type(size) is not int or size < 0):
        errors.append(f"{prefix}: byte_size must be null or a non-negative integer")
    for field in ("duplicate_of", "exclusion_reason"):
        value = obj.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(f"{prefix}: {field} must be a string or null")
    if obj.get("verification_status") == "excluded" and not str(obj.get("exclusion_reason") or "").strip():
        errors.append(f"{prefix}: excluded records require exclusion_reason")
    if obj.get("originality_status") == "original_verified" and not evidence:
        errors.append(f"{prefix}: original_verified records require provenance_evidence")
    if obj.get("notes") is not None and not isinstance(obj.get("notes"), str):
        errors.append(f"{prefix}: notes must be a string when present")
    return errors


def eligibility_blockers(record: dict) -> list[str]:
    """Return explicit reasons an item cannot count as a verified original sample."""
    blockers = []
    if record.get("verification_status") != "verified":
        blockers.append("verification_not_verified")
    if record.get("originality_status") != "original_verified":
        blockers.append("originality_not_verified")
    if record.get("provenance_confidence") != "high":
        blockers.append("provenance_not_high")
    if record.get("access_basis") not in ALLOWED_VERIFIED_ACCESS:
        blockers.append("rights_or_access_basis_not_verified")
    if record.get("usage_scope") not in {"full_text_analysis", "feature_analysis_only"}:
        blockers.append("usage_scope_does_not_allow_document_analysis")
    if record.get("access_status") != "accessed":
        blockers.append("document_content_not_accessed")
    if not record.get("license") or record.get("license", "").strip().lower() in {"unknown", "unclear", "n/a"}:
        blockers.append("license_or_permission_not_recorded")
    if not record.get("provenance_evidence"):
        blockers.append("provenance_evidence_missing")
    if record.get("privacy_review") not in {"passed", "not_required"}:
        blockers.append("privacy_review_not_passed")
    if record.get("anonymization_status") not in {"verified", "not_applicable"}:
        blockers.append("anonymization_not_verified")
    if record.get("duplicate_of"):
        blockers.append("declared_duplicate_or_version")
    if record.get("exclusion_reason"):
        blockers.append("exclusion_reason_present")
    if canonical_url(str(record.get("source_url", ""))) is None:
        blockers.append("source_url_invalid")
    return blockers


def load_manifest(path: Path, target: int = TARGET_DEFAULT) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc.msg}, column {exc.colno})")
            continue
        row_errors = validate_record(obj, line_no)
        if row_errors:
            errors.extend(row_errors)
            continue
        rows.append(dict(obj, _line_no=line_no))

    # A repeated stable ID is a manifest integrity error, not a new sample.
    id_owner: dict[str, int] = {}
    for row in rows:
        rid = row["record_id"]
        if rid in id_owner:
            errors.append(f"line {row['_line_no']}: duplicate record_id {rid!r} (first on line {id_owner[rid]})")
        else:
            id_owner[rid] = row["_line_no"]
    ids = {r["record_id"] for r in rows}
    for row in rows:
        ref = row.get("duplicate_of")
        if ref and ref not in ids:
            errors.append(f"line {row['_line_no']}: duplicate_of references unknown record_id {ref!r}")

    url_owner: dict[str, str] = {}
    hash_owner: dict[str, str] = {}
    first_seen_duplicates: dict[str, str] = {}
    per_row: list[dict] = []
    for row in rows:
        rid = row["record_id"]
        url = canonical_url(row["source_url"])
        digest = row.get("sha256")
        duplicate = row.get("duplicate_of") or None
        if url and url in url_owner and not duplicate:
            duplicate = url_owner[url]
            warnings.append(f"line {row['_line_no']}: canonical URL duplicates {duplicate!r}")
        elif url and url not in url_owner:
            url_owner[url] = rid
        if digest:
            digest = digest.lower()
            if digest in hash_owner and not duplicate:
                duplicate = hash_owner[digest]
                warnings.append(f"line {row['_line_no']}: content SHA-256 duplicates {duplicate!r}")
            elif digest not in hash_owner:
                hash_owner[digest] = rid
        blockers = eligibility_blockers(row)
        if duplicate:
            blockers.append("duplicate_url_hash_or_declared_version")
            first_seen_duplicates[rid] = duplicate
        per_row.append({
            "record_id": rid, "genre": row["genre"], "line": row["_line_no"],
            "eligible": not blockers, "blockers": sorted(set(blockers)),
            "duplicate_of_detected": duplicate,
            "source_family_id": row["source_family_id"],
        })

    eligible_ids = {item["record_id"] for item in per_row if item["eligible"]}
    by_genre = {}
    genre_total = Counter(row["genre"] for row in rows)
    genre_eligible = Counter(item["genre"] for item in per_row if item["eligible"])
    for genre in GENRES:
        by_genre[genre] = {"manifest_records": genre_total[genre], "eligible_unique": genre_eligible[genre]}
    originality = Counter(row["originality_status"] for row in rows)
    verification = Counter(row["verification_status"] for row in rows)
    excluded_reason_counts = Counter(
        blocker for item in per_row for blocker in item["blockers"]
    )
    eligible_count = len(eligible_ids)
    summary = {
        "schema_version": 1,
        "manifest_path": str(path),
        "target": target,
        "records_seen": sum(1 for line in raw_lines if line.strip()),
        "parseable_records": len(rows),
        "valid_manifest_records": len(rows) - sum(1 for e in errors if "duplicate record_id" in e or "duplicate_of references unknown" in e),
        "eligible_unique_original_documents": eligible_count,
        "target_remaining": max(0, target - eligible_count),
        "target_status": "COMPLETE" if eligible_count >= target and not errors else "INSUFFICIENT_EVIDENCE",
        "target_reached": eligible_count >= target and not errors,
        "originality_status_counts": dict(sorted(originality.items())),
        "verification_status_counts": dict(sorted(verification.items())),
        "by_genre": by_genre,
        "eligibility_blocker_counts": dict(sorted(excluded_reason_counts.items())),
        "errors": errors,
        "warnings": warnings,
        "items": per_row,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSONL manifest path")
    parser.add_argument("--target", type=int, default=TARGET_DEFAULT, help=f"verified unique documents required (default: {TARGET_DEFAULT})")
    parser.add_argument("--json", action="store_true", help="print the full machine-readable report")
    parser.add_argument("--require-target", action="store_true", help="exit non-zero if fewer than --target samples qualify")
    args = parser.parse_args(argv)
    if args.target < 1:
        parser.error("--target must be >= 1")
    try:
        report = load_manifest(args.manifest, args.target)
    except OSError as exc:
        print(f"cannot_read_manifest: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"status={report['target_status']}")
        print(f"verified_unique={report['eligible_unique_original_documents']}/{args.target}")
        print(f"records_seen={report['records_seen']} parseable={report['parseable_records']} errors={len(report['errors'])} warnings={len(report['warnings'])}")
        for genre, values in report["by_genre"].items():
            if values["manifest_records"]:
                print(f"genre.{genre}={values['eligible_unique']}/{values['manifest_records']}")
        for error in report["errors"]:
            print(f"ERROR {error}")
        for warning in report["warnings"]:
            print(f"WARNING {warning}")
    if report["errors"]:
        return 2
    if args.require_target and not report["target_reached"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
