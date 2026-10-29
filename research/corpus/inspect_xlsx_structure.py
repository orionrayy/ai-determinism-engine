#!/usr/bin/env python3
"""Inspect XLSX workbook structure without storing cell values or requiring third-party packages."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"m": MAIN_NS, "r": OFFICE_REL_NS, "p": PKG_REL_NS}


def _local_target(target: str) -> str:
    """Resolve an OOXML relationship target relative to xl/workbook.xml."""
    target = target.replace("\\", "/")
    if target.startswith("/"):
        return target.lstrip("/")
    normalized = str(PurePosixPath("xl") / target)
    parts = []
    for part in normalized.split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part not in {"", "."}:
            parts.append(part)
    return "/".join(parts)


def inspect_xlsx(path: Path, *, max_bytes: int = 20_000_000) -> dict:
    if not path.is_file():
        raise ValueError(f"not_a_file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"file_size_out_of_bounds: {size} bytes (max {max_bytes})")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not zipfile.is_zipfile(path):
        raise ValueError("not_a_valid_xlsx_zip_container")

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "xl/workbook.xml" not in names or "xl/_rels/workbook.xml.rels" not in names:
            raise ValueError("missing_workbook_xml_or_relationships")
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {}
        for rel in rels.findall("p:Relationship", NS):
            rel_id = rel.attrib.get("Id")
            target = rel.attrib.get("Target")
            rel_type = rel.attrib.get("Type", "")
            if rel_id and target and rel_type.endswith("/worksheet"):
                rel_targets[rel_id] = _local_target(target)

        sheets = []
        total_formula_cells = 0
        total_nonempty_cells = 0
        total_rows = 0
        total_merged_ranges = 0
        total_auto_filters = 0
        total_freeze_panes = 0
        cell_types = Counter()
        missing_sheets = []
        for sheet in workbook.findall(".//m:sheets/m:sheet", NS):
            name = sheet.attrib.get("name", "")
            state = sheet.attrib.get("state", "visible")
            rel_id = sheet.attrib.get(f"{{{OFFICE_REL_NS}}}id", "")
            target = rel_targets.get(rel_id)
            entry = {
                "name": name,
                "state": state,
                "relationship_id": rel_id,
                "worksheet_path": target,
                "dimension": None,
                "row_count": 0,
                "nonempty_cell_count": 0,
                "formula_cell_count": 0,
                "merged_range_count": 0,
                "auto_filter_present": False,
                "frozen_pane_present": False,
                "cell_type_counts": {},
            }
            if state not in {"visible", "hidden", "veryHidden"}:
                entry["state"] = "unknown"
            if not target or target not in names:
                missing_sheets.append(name)
                sheets.append(entry)
                continue
            root = ET.fromstring(archive.read(target))
            dim = root.find("m:dimension", NS)
            if dim is not None:
                entry["dimension"] = dim.attrib.get("ref")
            rows = root.findall(".//m:sheetData/m:row", NS)
            entry["row_count"] = len(rows)
            cells = root.findall(".//m:sheetData/m:row/m:c", NS)
            entry["nonempty_cell_count"] = len(cells)
            formulas = root.findall(".//m:sheetData/m:row/m:c/m:f", NS)
            entry["formula_cell_count"] = len({id(formula) for formula in formulas})
            merged = root.findall(".//m:mergeCells/m:mergeCell", NS)
            entry["merged_range_count"] = len(merged)
            entry["auto_filter_present"] = root.find("m:autoFilter", NS) is not None
            entry["frozen_pane_present"] = root.find(".//m:sheetViews/m:sheetView/m:pane", NS) is not None
            ct = Counter(cell.attrib.get("t", "n") for cell in cells)
            entry["cell_type_counts"] = dict(sorted(ct.items()))
            total_formula_cells += entry["formula_cell_count"]
            total_nonempty_cells += entry["nonempty_cell_count"]
            total_rows += entry["row_count"]
            total_merged_ranges += entry["merged_range_count"]
            total_auto_filters += int(entry["auto_filter_present"])
            total_freeze_panes += int(entry["frozen_pane_present"])
            cell_types.update(ct)
            sheets.append(entry)

        if not sheets:
            raise ValueError("no_worksheets_found")
        if missing_sheets:
            raise ValueError("missing_worksheet_parts: " + ", ".join(missing_sheets))

        styles_summary = {}
        if "xl/styles.xml" in names:
            styles = ET.fromstring(archive.read("xl/styles.xml"))
            for tag in ("numFmts", "fonts", "fills", "borders", "cellStyleXfs", "cellXfs", "cellStyles", "dxfs"):
                elem = styles.find(f"m:{tag}", NS)
                styles_summary[tag] = len(list(elem)) if elem is not None else 0

        defined_names = workbook.findall(".//m:definedNames/m:definedName", NS)
        tables = sorted(name for name in names if re.fullmatch(r"xl/tables/table\d+\.xml", name))
        result = {
            "status": "inspected",
            "file_name": path.name,
            "sha256": digest,
            "byte_size": size,
            "workbook_sheet_count": len(sheets),
            "visible_sheet_count": sum(s["state"] == "visible" for s in sheets),
            "hidden_sheet_count": sum(s["state"] in {"hidden", "veryHidden"} for s in sheets),
            "worksheet_names_and_structure": sheets,
            "totals": {
                "row_elements": total_rows,
                "nonempty_cells": total_nonempty_cells,
                "formula_cells": total_formula_cells,
                "merged_ranges": total_merged_ranges,
                "auto_filter_sheets": total_auto_filters,
                "sheets_with_frozen_panes": total_freeze_panes,
                "cell_types": dict(sorted(cell_types.items())),
                "defined_names": len(defined_names),
                "table_parts": len(tables),
            },
            "styles_summary": styles_summary,
            "table_parts": tables,
            "value_retention": "no cell values, formulas, or personal data are emitted; only counts and sheet structure",
        }
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-bytes", type=int, default=20_000_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_bytes < 1:
        parser.error("--max-bytes must be positive")
    try:
        result = inspect_xlsx(args.path, max_bytes=args.max_bytes)
    except (OSError, zipfile.BadZipFile, ET.ParseError, ValueError) as exc:
        print(f"xlsx_inspection_failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"status={result['status']}")
        print(f"sha256={result['sha256']}")
        print(f"byte_size={result['byte_size']}")
        print(f"workbook_sheet_count={result['workbook_sheet_count']}")
        print(f"formula_cells={result['totals']['formula_cells']}")
        for sheet in result["worksheet_names_and_structure"]:
            print(f"sheet={sheet['name']!r}\tstate={sheet['state']}\tdimension={sheet['dimension']}\tcells={sheet['nonempty_cell_count']}\tformulas={sheet['formula_cell_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
