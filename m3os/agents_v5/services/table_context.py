"""Full table context assembly for M3OS Agents v5.

This module reads user-provided CSV/Excel files and formats their full table
contents as additional context before agent creation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


@dataclass(frozen=True)
class TableSource:
    """Loaded table content from one CSV file or one Excel sheet."""

    path: str
    source_type: str
    dataframe: pd.DataFrame
    sheet_name: Optional[str] = None


def combine_context_blocks(*blocks: Optional[str]) -> Optional[str]:
    """Combine non-empty context blocks into one additional context string."""
    non_empty = [block.strip() for block in blocks if block and block.strip()]
    if not non_empty:
        return None
    return "\n\n".join(non_empty)


def read_table_files(paths: Sequence[str]) -> List[TableSource]:
    """Read CSV/Excel files from the provided paths.

    Missing, empty, duplicate, unsupported, or unreadable files are skipped with
    a console warning because table context is auxiliary and should not prevent
    the main optimization workflow.
    """
    loaded: List[TableSource] = []
    seen: set[str] = set()
    candidate_paths = [str(paths)] if isinstance(paths, (str, Path)) else paths

    for raw_path in candidate_paths:
        path_text = str(raw_path).strip()
        if not path_text:
            continue

        path = Path(path_text).expanduser()
        dedupe_key = str(path.resolve()) if path.exists() else str(path)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        suffix = path.suffix.lower()
        if suffix == ".csv":
            loaded.extend(_read_csv_file(path))
        elif suffix == ".tsv":
            loaded.extend(_read_tsv_file(path))
        elif suffix in {".xlsx", ".xlsm", ".xls"}:
            loaded.extend(_read_excel_file(path))
        else:
            print(f"Unsupported table file format; skipping: {path}")

    return loaded


async def build_table_context(
    table_paths: Optional[Sequence[str]],
) -> Optional[str]:
    """Build a full table context block for all agents."""
    if not table_paths:
        return None

    loaded_sources = read_table_files(table_paths)
    if not loaded_sources:
        return None

    print("\n=== Building full table context ===")
    print(f"Loaded table sources: {len(loaded_sources)}")
    for source in loaded_sources:
        sheet_info = f", sheet: {source.sheet_name}" if source.sheet_name else ""
        print(
            f"  - Added full table source: {source.path}{sheet_info} "
            f"({len(source.dataframe)} rows, {len(source.dataframe.columns)} columns)"
        )

    return _format_full_table_context(loaded_sources)


def _read_csv_file(path: Path) -> List[TableSource]:
    try:
        dataframe = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Table file not found; skipping: {path}")
        return []
    except pd.errors.EmptyDataError:
        print(f"Table file is empty; skipping: {path}")
        return []
    except Exception as exc:
        print(f"Failed to read CSV table; skipping: {path} ({exc})")
        return []

    if _is_empty_table(dataframe):
        print(f"Table file is empty; skipping: {path}")
        return []

    return [TableSource(path=str(path), source_type="CSV", dataframe=dataframe)]


def _read_tsv_file(path: Path) -> List[TableSource]:
    try:
        dataframe = pd.read_csv(path, sep="\t")
    except FileNotFoundError:
        print(f"Table file not found; skipping: {path}")
        return []
    except pd.errors.EmptyDataError:
        print(f"Table file is empty; skipping: {path}")
        return []
    except Exception as exc:
        print(f"Failed to read TSV table; skipping: {path} ({exc})")
        return []

    if _is_empty_table(dataframe):
        print(f"Table file is empty; skipping: {path}")
        return []

    return [TableSource(path=str(path), source_type="TSV", dataframe=dataframe)]


def _read_excel_file(path: Path) -> List[TableSource]:
    try:
        workbook = pd.read_excel(path, sheet_name=None)
    except FileNotFoundError:
        print(f"Table file not found; skipping: {path}")
        return []
    except Exception as exc:
        print(f"Failed to read Excel table; skipping: {path} ({exc})")
        return []

    loaded: List[TableSource] = []
    for sheet_name, dataframe in workbook.items():
        if _is_empty_table(dataframe):
            print(f"Excel sheet is empty; skipping: {path} [{sheet_name}]")
            continue
        loaded.append(
            TableSource(
                path=str(path),
                source_type="Excel",
                dataframe=dataframe,
                sheet_name=str(sheet_name),
            )
        )

    if not loaded:
        print(f"Excel table has no usable sheets; skipping: {path}")

    return loaded


def _is_empty_table(dataframe: pd.DataFrame) -> bool:
    return dataframe.empty and len(dataframe.columns) == 0


def _format_full_table_context(loaded_sources: Sequence[TableSource]) -> str:
    """Format full CSV/Excel table content for system prompt injection."""
    lines: List[str] = [
        "[FULL USER-PROVIDED TABLE CONTEXT]",
        "The following CSV/Excel table sources are provided in full as task-specific context.",
        "Study each Table Source independently, keep the source and sheet boundaries clear, and use table values only when they are directly applicable to the user's optimization task.",
        "When proposing, evaluating, or reporting molecule designs, reference the relevant Table Source number when a table-supported value or trend influences the reasoning.",
        "Do not invent table values, columns, measurements, SAR, ADMET, PK, or experimental conclusions that are not supported by these sources. Do not overrule explicit user constraints.",
    ]

    for index, source in enumerate(loaded_sources, start=1):
        dataframe = _normalize_dataframe_for_markdown(source.dataframe)
        column_names = [str(column) for column in dataframe.columns]
        lines.extend(
            [
                "",
                "=" * 80,
                f"## Table Source {index}",
                f"Path: {source.path}",
                f"Format: {source.source_type}",
                f"Sheet: {source.sheet_name}" if source.sheet_name else "Sheet: N/A",
                f"Rows: {len(dataframe)}",
                f"Columns: {len(column_names)}",
                f"Column names: {', '.join(column_names)}",
                "--- BEGIN TABLE MARKDOWN ---",
                dataframe.to_markdown(index=False),
                "--- END TABLE MARKDOWN ---",
            ]
        )

    return "\n".join(lines)


def _normalize_dataframe_for_markdown(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert missing values to blanks for cleaner prompt tables."""
    return dataframe.astype(object).where(pd.notna(dataframe), "")
