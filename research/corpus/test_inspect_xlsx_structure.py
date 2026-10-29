import tempfile
import unittest
import zipfile
from pathlib import Path

import inspect_xlsx_structure as inspector


WORKBOOK = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <sheets>
  <sheet name="Header" sheetId="1" r:id="rId1"/>
  <sheet name="Data" sheetId="2" state="hidden" r:id="rId2"/>
 </sheets>
 <definedNames><definedName name="Print_Area">Data!$A$1:$B$2</definedName></definedNames>
</workbook>"""
RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>"""
SHEET1 = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <dimension ref="A1:B2"/>
 <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
 <sheetData>
  <row r="1"><c r="A1" t="inlineStr"><is><t>metadata label</t></is></c><c r="B1" t="inlineStr"><is><t>metadata value</t></is></c></row>
  <row r="2"><c r="A2" t="n"><v>1</v></c><c r="B2" t="n"><v>2</v></c></row>
 </sheetData>
 <mergeCells count="1"><mergeCell ref="A1:B1"/></mergeCells>
 <autoFilter ref="A1:B2"/>
</worksheet>"""
SHEET2 = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <dimension ref="A1:B3"/>
 <sheetData>
  <row r="1"><c r="A1" t="inlineStr"><is><t>time</t></is></c><c r="B1" t="inlineStr"><is><t>voltage</t></is></c></row>
  <row r="2"><c r="A2" t="n"><v>0</v></c><c r="B2"><f>B3*2</f><v>4</v></c></row>
  <row r="3"><c r="A3" t="n"><v>1</v></c><c r="B3"><f>B2+1</f><v>2</v></c></row>
 </sheetData>
</worksheet>"""
STYLES = """<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <numFmts/><fonts/><fills/><borders/><cellStyleXfs/><cellXfs/><cellStyles/><dxfs/>
</styleSheet>"""


class XlsxStructureTests(unittest.TestCase):
    def make_xlsx(self, folder: Path) -> Path:
        path = folder / "sample.xlsx"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("xl/workbook.xml", WORKBOOK)
            zf.writestr("xl/_rels/workbook.xml.rels", RELS)
            zf.writestr("xl/worksheets/sheet1.xml", SHEET1)
            zf.writestr("xl/worksheets/sheet2.xml", SHEET2)
            zf.writestr("xl/styles.xml", STYLES)
        return path

    def test_sheet_and_formula_structure_are_reported_without_cell_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = inspector.inspect_xlsx(self.make_xlsx(Path(tmp)))
        self.assertEqual(result["status"], "inspected")
        self.assertEqual(result["workbook_sheet_count"], 2)
        self.assertEqual(result["hidden_sheet_count"], 1)
        self.assertEqual(result["totals"]["formula_cells"], 2)
        self.assertEqual(result["totals"]["merged_ranges"], 1)
        self.assertEqual(result["totals"]["auto_filter_sheets"], 1)
        self.assertEqual(result["totals"]["sheets_with_frozen_panes"], 1)
        self.assertEqual(result["worksheet_names_and_structure"][1]["name"], "Data")
        self.assertNotIn("cell_values", result)
        self.assertNotIn("formulas", result)

    def test_non_zip_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.xlsx"
            path.write_bytes(b"not a zip file")
            with self.assertRaisesRegex(ValueError, "not_a_valid_xlsx_zip_container"):
                inspector.inspect_xlsx(path)

    def test_file_size_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_xlsx(Path(tmp))
            with self.assertRaisesRegex(ValueError, "file_size_out_of_bounds"):
                inspector.inspect_xlsx(path, max_bytes=10)

    def test_missing_sheet_part_is_rejected(self):
        broken_rels = RELS.replace('Target="worksheets/sheet2.xml"', 'Target="worksheets/not-found.xml"')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.xlsx"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("xl/workbook.xml", WORKBOOK)
                zf.writestr("xl/_rels/workbook.xml.rels", broken_rels)
                zf.writestr("xl/worksheets/sheet1.xml", SHEET1)
            with self.assertRaisesRegex(ValueError, "missing_worksheet_parts"):
                inspector.inspect_xlsx(path)


if __name__ == "__main__":
    unittest.main()
