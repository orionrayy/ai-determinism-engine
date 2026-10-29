#!/usr/bin/env python3
"""Check traceability and limitations for document-feature analysis JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import validate_corpus as vc


FEATURE_REQUIRED = {
    "record_id", "source_family_id", "analysis_version", "reviewed_at",
    "evidence_urls", "method", "observed_structure", "review_flags",
}
METHOD_REQUIRED = {"access", "text_review", "visual_review", "stored_content"}
REVIEW_REQUIRED = {
    "source_is_authored_publication", "factual_claims_independently_verified",
    "benchmark_role", "generalization_status",
}
OBSERVATION_REQUIRED = {"front_matter", "visual_features"}


def _is_http_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parts = urlsplit(value)
        return parts.scheme in {"http", "https"} and bool(parts.hostname) and not parts.username and not parts.password
    except ValueError:
        return False


def load_features(features_path: Path, manifest_path: Path, target: int = 500) -> dict:
    corpus = vc.load_manifest(manifest_path, target)
    manifest_by_id = {}
    manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    for line_no, raw in enumerate(manifest_lines, start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("record_id"), str):
            manifest_by_id.setdefault(obj["record_id"], obj)

    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict] = []
    lines = features_path.read_text(encoding="utf-8").splitlines()
    seen: dict[str, int] = {}
    for line_no, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
            continue
        if not isinstance(item, dict):
            errors.append(f"line {line_no}: feature row must be an object")
            continue
        missing = sorted(FEATURE_REQUIRED - set(item))
        if missing:
            errors.append(f"line {line_no}: missing required fields: {', '.join(missing)}")
        rid = item.get("record_id")
        if not isinstance(rid, str) or not rid.strip():
            errors.append(f"line {line_no}: record_id must be a non-empty string")
            continue
        if rid in seen:
            errors.append(f"line {line_no}: duplicate analysis for record_id {rid!r} (first line {seen[rid]})")
            continue
        seen[rid] = line_no
        manifest_record = manifest_by_id.get(rid)
        if manifest_record is None:
            errors.append(f"line {line_no}: record_id {rid!r} does not exist in the corpus manifest")
            continue
        if item.get("source_family_id") != manifest_record.get("source_family_id"):
            errors.append(f"line {line_no}: source_family_id does not match manifest for {rid!r}")
        if not isinstance(item.get("analysis_version"), str) or not item.get("analysis_version").strip():
            errors.append(f"line {line_no}: analysis_version must be non-empty")
        if not isinstance(item.get("reviewed_at"), str) or not item.get("reviewed_at").strip():
            errors.append(f"line {line_no}: reviewed_at must be non-empty")

        evidence_urls = item.get("evidence_urls")
        allowed_evidence = set(manifest_record.get("provenance_evidence") or [])
        if not isinstance(evidence_urls, list) or not evidence_urls:
            errors.append(f"line {line_no}: evidence_urls must be a non-empty array")
        else:
            for url in evidence_urls:
                if not _is_http_url(url):
                    errors.append(f"line {line_no}: invalid evidence URL for {rid!r}")
                elif url not in allowed_evidence and url != manifest_record.get("source_url"):
                    errors.append(f"line {line_no}: evidence URL not recorded in manifest provenance for {rid!r}")

        method = item.get("method")
        if not isinstance(method, dict):
            errors.append(f"line {line_no}: method must be an object")
        else:
            for field in sorted(METHOD_REQUIRED - set(method)):
                errors.append(f"line {line_no}: method.{field} is required")
            if isinstance(method.get("stored_content"), str):
                stored = method["stored_content"].lower()
                if any(term in stored for term in ("full text stored", "full-text stored", "pdf stored", "full text retained")):
                    warnings.append(f"line {line_no}: stored_content claims full text retention; inspect rights/privacy manually")

        observed = item.get("observed_structure")
        if not isinstance(observed, dict) or not observed:
            errors.append(f"line {line_no}: observed_structure must be a non-empty object")
        else:
            for field in sorted(OBSERVATION_REQUIRED - set(observed)):
                errors.append(f"line {line_no}: observed_structure.{field} is required")
            for field, value in observed.items():
                if not isinstance(value, (str, list, dict)):
                    errors.append(f"line {line_no}: observed_structure.{field} must be descriptive text/list/object")
                elif isinstance(value, list) and any(not isinstance(v, str) or not v.strip() for v in value):
                    errors.append(f"line {line_no}: observed_structure.{field} has an empty/non-text item")

        review = item.get("review_flags")
        if not isinstance(review, dict):
            errors.append(f"line {line_no}: review_flags must be an object")
        else:
            for field in sorted(REVIEW_REQUIRED - set(review)):
                errors.append(f"line {line_no}: review_flags.{field} is required")
            if review.get("factual_claims_independently_verified") is False:
                warnings.append(f"line {line_no}: factual claims not independently verified for {rid!r}")
            if review.get("generalization_status") and "single sample" in str(review["generalization_status"]).lower():
                warnings.append(f"line {line_no}: feature record explicitly limits generalization for {rid!r}")

        if not vc.eligibility_blockers(manifest_record):
            pass
        else:
            warnings.append(f"line {line_no}: feature analysis belongs to a manifest record that is not currently corpus-eligible: {rid!r}")

        records.append({
            "record_id": rid,
            "source_family_id": item.get("source_family_id"),
            "genre": manifest_record.get("genre"),
            "line": line_no,
        })

    feature_ids = {r["record_id"] for r in records}
    corpus_ids = {r["record_id"] for r in corpus.get("items", []) if r["eligible"]}
    eligible_with_features = corpus_ids & feature_ids
    eligible_without_features = sorted(corpus_ids - feature_ids)
    if eligible_without_features:
        warnings.append(
            "eligible manifest records without feature analysis: " + ", ".join(eligible_without_features)
        )
    by_genre_manifest = {genre: data["eligible_unique"] for genre, data in corpus["by_genre"].items()}
    by_genre_features = dict(sorted(Counter(r["genre"] for r in records if r["record_id"] in corpus_ids).items()))
    summary = {
        "manifest_path": str(manifest_path),
        "features_path": str(features_path),
        "target": target,
        "target_status": corpus["target_status"],
        "eligible_unique_documents": corpus["eligible_unique_original_documents"],
        "analysis_records": len(records),
        "eligible_documents_with_analysis": len(eligible_with_features),
        "eligible_documents_missing_analysis": eligible_without_features,
        "manifest_eligible_by_genre": by_genre_manifest,
        "features_by_eligible_genre": by_genre_features,
        "errors": errors,
        "warnings": warnings,
        "feature_records": records,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("features", type=Path)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.target < 1:
        parser.error("--target must be >= 1")
    try:
        report = load_features(args.features, args.manifest, args.target)
    except OSError as exc:
        print(f"cannot_read_input: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"target_status={report['target_status']}")
        print(f"eligible_documents={report['eligible_unique_documents']}/{args.target}")
        print(f"analysis_records={report['analysis_records']}")
        print(f"analysis_coverage={report['eligible_documents_with_analysis']}/{report['eligible_unique_documents']}")
        for error in report["errors"]:
            print(f"ERROR {error}")
        for warning in report["warnings"]:
            print(f"WARNING {warning}")
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
