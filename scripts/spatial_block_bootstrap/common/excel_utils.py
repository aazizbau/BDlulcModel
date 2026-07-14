"""Write simple multi-sheet XLSX workbooks without optional Excel packages."""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


XML_HEADER = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RELATIONSHIP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def column_name(index: int) -> str:
    """Convert a zero-based column index to an Excel column name."""
    result = ""
    current = index + 1
    while current:
        current, remainder = divmod(current - 1, 26)
        result = chr(65 + remainder) + result
    return result


def cell_xml(reference: str, value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return f'<c r="{reference}"/>'
    if isinstance(value, (bool, np.bool_)):
        return f'<c r="{reference}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            return f'<c r="{reference}"/>'
        return f'<c r="{reference}"><v>{value}</v></c>'
    text = escape(str(value))
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return f'<c r="{reference}" t="inlineStr"><is><t{preserve}>{text}</t></is></c>'


def worksheet_xml(frame: pd.DataFrame) -> str:
    rows = [list(frame.columns), *frame.itertuples(index=False, name=None)]
    xml_rows = []
    for row_number, values in enumerate(rows, start=1):
        cells = "".join(
            cell_xml(f"{column_name(column_index)}{row_number}", value)
            for column_index, value in enumerate(values)
        )
        xml_rows.append(f'<row r="{row_number}">{cells}</row>')
    return (
        XML_HEADER
        + f'<worksheet xmlns="{SPREADSHEET_NS}"><sheetData>'
        + "".join(xml_rows)
        + "</sheetData></worksheet>"
    )


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """Write DataFrames as an XLSX workbook using inline worksheet strings."""
    if not sheets:
        raise ValueError("At least one worksheet is required.")
    path.parent.mkdir(parents=True, exist_ok=True)

    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    ]
    workbook_sheets = []
    workbook_relationships = []
    for index, sheet_name in enumerate(sheets, start=1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        workbook_sheets.append(
            f'<sheet name="{escape(sheet_name)}" sheetId="{index}" r:id="rId{index}"/>'
        )
        workbook_relationships.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )

    content_types = (
        XML_HEADER
        + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        + '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        + '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + "</Types>"
    )
    root_relationships = (
        XML_HEADER
        + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + '<Relationship Id="rId1" '
        + 'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        + 'Target="xl/workbook.xml"/>'
        + "</Relationships>"
    )
    workbook = (
        XML_HEADER
        + f'<workbook xmlns="{SPREADSHEET_NS}" xmlns:r="{RELATIONSHIP_NS}"><sheets>'
        + "".join(workbook_sheets)
        + "</sheets></workbook>"
    )
    workbook_rels = (
        XML_HEADER
        + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(workbook_relationships)
        + "</Relationships>"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        for index, frame in enumerate(sheets.values(), start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", worksheet_xml(frame))
