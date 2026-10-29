#!/usr/bin/env python3
"""Harvest bounded, public DOAJ OAI-PMH article metadata into discovery candidates.

This tool stores CC0 metadata candidates only. It does not download article PDFs,
retain abstracts, infer article licenses, or create corpus manifest entries.
See https://doaj.org/terms/ and https://doaj.org/docs/oai-pmh/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Iterable

OAI_ENDPOINT = "https://doaj.org/oai.article"
OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"
DC_NS = "{http://purl.org/dc/elements/1.1/}"
OAI_DC_NS = "{http://www.openarchives.org/OAI/2.0/oai_dc/}"
DEFAULT_REGISTRY = Path(__file__).with_name("source_registry.json")
DEFAULT_OUTPUT = Path(__file__).with_name("candidates.jsonl")
DEFAULT_ISSNS = (
    "2548-592X", "2656-4602", "2686-116X", "2579-9320", "2614-4948",
    "2550-0538", "2723-8067", "2656-4610", "2461-0747", "2829-5943",
)
ISSN_RE = re.compile(r"(?<!\d)(\d{4})-?(\d{3}[\dXx])(?!\d)")
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_CANDIDATE_TEXT = 1000


class HarvestError(RuntimeError):
    """Raised for malformed metadata or a failed bounded request."""


def normalize_issn(value: str) -> str | None:
    match = ISSN_RE.fullmatch(value.strip())
    if not match:
        return None
    first_seven = match.group(1) + match.group(2)[:3]
    check_char = match.group(2)[3].upper()
    weighted_sum = sum(int(digit) * weight for digit, weight in zip(first_seven, range(8, 1, -1)))
    expected = (11 - (weighted_sum % 11)) % 11
    expected_char = "X" if expected == 10 else str(expected)
    if check_char != expected_char:
        return None
    return match.group(1) + "-" + match.group(2).upper()


def _issns_in_values(values: Iterable[str]) -> set[str]:
    found: set[str] = set()
    for value in values:
        stripped = value.strip()
        direct = normalize_issn(stripped)
        if direct:
            found.add(direct)
            continue
        labeled = re.fullmatch(r"(?i)ISSN\\s*[:=]?\\s*([0-9]{4}-?[0-9]{3}[0-9X])", stripped)
        if labeled:
            normalized = normalize_issn(labeled.group(1))
            if normalized:
                found.add(normalized)
            continue
        # Some catalogues identify the journal by its canonical /toc/<ISSN> URL.
        toc = re.search(r"/toc/([0-9]{4}-?[0-9]{3}[0-9X])/?$", stripped, re.IGNORECASE)
        if toc:
            normalized = normalize_issn(toc.group(1))
            if normalized:
                found.add(normalized)
    return found


def _first_year(values: Iterable[str]) -> int | None:
    for value in values:
        match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2}|2100)\b", value)
        if match:
            year = int(match.group(1))
            if 1800 <= year <= date.today().year + 1:
                return year
    return None


def _first_doi(values: Iterable[str]) -> str | None:
    for value in values:
        match = DOI_RE.search(value)
        if match:
            return match.group(0).rstrip(".,;:)]}").lower()
    return None


def _registry_issns(registry_path: Path) -> set[str]:
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarvestError(f"cannot read source registry: {exc}") from exc
    found: set[str] = set()
    for item in data.get("sources", []):
        url = str(item.get("source_url", ""))
        match = re.search(r"/toc/([0-9]{4}-?[0-9]{3}[0-9Xx])/?$", url)
        if match:
            normalized = normalize_issn(match.group(1))
            if normalized:
                found.add(normalized)
    return found


def build_request_url(params: dict[str, str]) -> str:
    return OAI_ENDPOINT + "?" + urllib.parse.urlencode(params)


def fetch_xml(url: str, *, timeout: float, max_bytes: int = MAX_RESPONSE_BYTES) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ai-determinism-engine-hallmark-corpus/1.0 (metadata discovery; contact via GitHub repository)",
            "Accept": "application/xml, text/xml;q=0.9, */*;q=0.1",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(max_bytes + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HarvestError(f"DOAJ metadata request failed: {type(exc).__name__}") from exc
    if len(data) > max_bytes:
        raise HarvestError(f"DOAJ response exceeds configured byte limit ({max_bytes})")
    return data


def _texts(element: ET.Element, local_name: str, namespace: str = DC_NS) -> list[str]:
    return [node.text.strip() for node in element.findall(f"{namespace}{local_name}")
            if node.text and node.text.strip()]


def _text(element: ET.Element, local_name: str, namespace: str = DC_NS) -> str | None:
    values = _texts(element, local_name, namespace)
    return values[0][:MAX_CANDIDATE_TEXT] if values else None


def parse_records(xml_bytes: bytes, target_issns: set[str]) -> tuple[list[dict], str | None, dict]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise HarvestError(f"invalid OAI XML: {exc}") from exc

    error = root.find(f".//{OAI_NS}error")
    if error is not None:
        code = error.attrib.get("code", "unknown")
        message = (error.text or "").strip()
        if code == "noRecordsMatch":
            return [], None, {"oai_error": code, "message": message}
        raise HarvestError(f"DOAJ OAI error {code}: {message[:250]}")

    records: list[dict] = []
    for record in root.findall(f".//{OAI_NS}record"):
        header = record.find(f"{OAI_NS}header")
        if header is None or header.attrib.get("status") == "deleted":
            continue
        oai_id_el = header.find(f"{OAI_NS}identifier")
        oai_id = (oai_id_el.text or "").strip() if oai_id_el is not None else ""
        metadata = record.find(f"{OAI_NS}metadata/{OAI_DC_NS}dc")
        if metadata is None:
            continue

        titles = _texts(metadata, "title")
        identifiers = _texts(metadata, "identifier")
        relations = _texts(metadata, "relation")
        creators = _texts(metadata, "creator")
        dates = _texts(metadata, "date")
        publishers = _texts(metadata, "publisher")
        languages = _texts(metadata, "language")
        subjects = _texts(metadata, "subject")
        types = _texts(metadata, "type")
        all_ids = identifiers + relations + [oai_id]
        record_issns = _issns_in_values(all_ids)
        matching_issns = sorted(record_issns & target_issns)
        if not matching_issns:
            continue

        title = " ".join(titles).strip()[:MAX_CANDIDATE_TEXT]
        if not title:
            continue
        doi = _first_doi(identifiers + relations)
        canonical_identifier = doi or oai_id or next((v for v in identifiers if v.startswith("http")), "")
        if not canonical_identifier:
            canonical_identifier = "|".join([title, "|".join(creators), "|".join(dates)])
        candidate_id = "doaj-" + hashlib.sha256(canonical_identifier.strip().lower().encode("utf-8")).hexdigest()[:24]
        source_url = next((v.strip() for v in identifiers if v.strip().startswith("https://doaj.org/article/")), None)
        if not source_url:
            source_url = next((v.strip() for v in identifiers if v.strip().startswith("http")), None)
        records.append({
            "candidate_id": candidate_id,
            "source": "DOAJ OAI-PMH article metadata",
            "metadata_license": "CC0 (DOAJ metadata terms)",
            "metadata_license_evidence": "https://doaj.org/terms/",
            "oai_document_identifier": oai_id or None,
            "title": title,
            "creators": creators[:20],
            "publisher": " ".join(publishers)[:MAX_CANDIDATE_TEXT] or None,
            "publication_year": _first_year(dates),
            "doi": doi,
            "source_url": source_url,
            "full_text_links": [v[:2000] for v in relations if v.startswith(("https://", "http://"))][:10],
            "journal_issns": matching_issns,
            "language_metadata": languages[:10],
            "subjects": subjects[:30],
            "dc_type": types[:5],
            "content_license_status": "not_verified_for_this_article",
            "content_access_status": "not_checked",
            "originality_status": "candidate_only",
            "provenance_confidence": "metadata_discovery_only",
            "corpus_eligible": False,
        })

    token_element = root.find(f".//{OAI_NS}resumptionToken")
    token = (token_element.text or "").strip() if token_element is not None else ""
    metadata = {
        "records_in_response": len(root.findall(f".//{OAI_NS}record")),
        "matching_candidates": len(records),
        "has_resumption_token": bool(token),
    }
    return records, token or None, metadata


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n",
                                         dir=str(path.parent), prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            tmp_name = handle.name
            handle.write(payload)
            handle.flush()
        Path(tmp_name).replace(path)
    finally:
        if tmp_name and Path(tmp_name).exists():
            Path(tmp_name).unlink()


def load_existing(path: Path) -> dict[str, dict]:
    existing: dict[str, dict] = {}
    if not path.exists():
        return existing
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HarvestError(f"existing candidate manifest has invalid JSON at line {line_no}") from exc
        candidate_id = item.get("candidate_id") if isinstance(item, dict) else None
        if not isinstance(candidate_id, str) or not candidate_id:
            raise HarvestError(f"existing candidate manifest has no candidate_id at line {line_no}")
        existing[candidate_id] = item
    return existing


def harvest(
    *,
    output_path: Path,
    target_issns: set[str],
    max_pages: int = 2,
    max_records: int = 200,
    timeout: float = 20.0,
    min_interval: float = 1.0,
    from_date: str | None = None,
    until_date: str | None = None,
    dry_run: bool = False,
) -> dict:
    if not target_issns:
        raise HarvestError("no target ISSNs configured")
    if max_pages < 1 or max_records < 1 or timeout <= 0 or min_interval < 0:
        raise HarvestError("max-pages/max-records/timeout must be positive and min-interval non-negative")
    for date_value in (from_date, until_date):
        if date_value:
            try:
                date.fromisoformat(date_value)
            except ValueError as exc:
                raise HarvestError("from/until dates must use YYYY-MM-DD") from exc
    if from_date and until_date and from_date > until_date:
        raise HarvestError("--from must be on or before --until")

    existing = load_existing(output_path)
    added: dict[str, dict] = {}
    token: str | None = None
    seen_tokens: set[str] = set()
    pages = 0
    source_records = 0
    matched = 0
    request_errors: list[str] = []
    last_request_at = 0.0
    stop_reason = "max_pages_reached"

    while pages < max_pages and source_records < max_records:
        if last_request_at:
            wait = min_interval - (time.monotonic() - last_request_at)
            if wait > 0:
                time.sleep(wait)
        if token:
            params = {"verb": "ListRecords", "resumptionToken": token}
        else:
            params = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}
            if from_date:
                params["from"] = from_date
            if until_date:
                params["until"] = until_date
        url = build_request_url(params)
        last_request_at = time.monotonic()
        xml_bytes = fetch_xml(url, timeout=timeout)
        page_rows, next_token, response_meta = parse_records(xml_bytes, target_issns)
        pages += 1
        source_records += int(response_meta.get("records_in_response", 0))
        matched += len(page_rows)
        for row in page_rows:
            added[row["candidate_id"]] = row
        if response_meta.get("oai_error") == "noRecordsMatch":
            stop_reason = "no_records_match"
            break
        if not next_token:
            stop_reason = "feed_exhausted"
            break
        if next_token in seen_tokens:
            raise HarvestError("repeated OAI resumption token; aborting to prevent an infinite loop")
        seen_tokens.add(next_token)
        token = next_token

    merged = dict(existing)
    merged.update(added)
    merged_rows = sorted(merged.values(), key=lambda row: (row.get("source", ""), row.get("candidate_id", "")))
    if not dry_run:
        _atomic_write_jsonl(output_path, merged_rows)
    return {
        "status": "CANDIDATES_DISCOVERED" if added else "NO_NEW_CANDIDATES",
        "source": "DOAJ OAI-PMH",
        "metadata_license": "CC0 (DOAJ metadata terms)",
        "content_rights_warning": "Article-content license and actual full-text accessibility must be checked per item; metadata candidates do not qualify as corpus samples.",
        "output_path": str(output_path),
        "dry_run": dry_run,
        "target_issns": sorted(target_issns),
        "pages_requested": pages,
        "source_records_scanned": source_records,
        "matching_candidate_records_in_run": matched,
        "new_unique_candidates": len(added.keys() - existing.keys()),
        "candidate_inventory_unique": len(merged_rows),
        "stop_reason": stop_reason,
        "target_500_status": "INSUFFICIENT_EVIDENCE",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--issn", action="append", default=[], help="target journal ISSN; can be repeated")
    parser.add_argument("--max-pages", type=int, default=2, help="hard request/page cap (default: 2)")
    parser.add_argument("--max-records", type=int, default=200, help="hard source-record processing cap (default: 200)")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--min-interval", type=float, default=1.0, help="minimum seconds between requests")
    parser.add_argument("--from", dest="from_date", help="optional inclusive OAI date YYYY-MM-DD")
    parser.add_argument("--until", dest="until_date", help="optional inclusive OAI date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="print report and do not write candidate file")
    parser.add_argument("--json", action="store_true", help="emit machine-readable report")
    args = parser.parse_args(argv)

    try:
        registry_issns = _registry_issns(args.registry)
        requested = {normalize_issn(v) for v in args.issn}
        if any(value is None for value in requested):
            raise HarvestError("one or more --issn values are invalid")
        target_issns = {value for value in requested if value} if requested else (registry_issns or set(DEFAULT_ISSNS))
        report = harvest(
            output_path=args.output, target_issns=target_issns,
            max_pages=args.max_pages, max_records=args.max_records,
            timeout=args.timeout, min_interval=args.min_interval,
            from_date=args.from_date, until_date=args.until_date,
            dry_run=args.dry_run,
        )
    except HarvestError as exc:
        print(f"harvest_error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"status={report['status']}")
        print(f"pages_requested={report['pages_requested']} source_records_scanned={report['source_records_scanned']}")
        print(f"new_unique_candidates={report['new_unique_candidates']} total_candidate_inventory={report['candidate_inventory_unique']}")
        print(f"stop_reason={report['stop_reason']}")
        print(report["content_rights_warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
