"""SMILES validation helpers for agents_v5 molecular optimization."""

from __future__ import annotations

import re
import logging
from typing import Any

import pandas as pd

try:  # pragma: no cover - RDKit may be unavailable in lightweight envs
    from rdkit import Chem
    from rdkit import rdBase
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover
    Chem = None
    rdBase = None
    DataStructs = None
    AllChem = None

NON_SMILES_PLACEHOLDERS = {
    "<unknown>",
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "not available",
    "not provided",
    "missing",
}

NON_SMILES_PROPERTY_NAMES = {
    "caco-2",
    "caco2",
    "p-gp",
    "pgp",
    "bbb",
    "bbbp",
    "hia",
    "herg",
    "logp",
    "logd",
    "clogp",
    "tpsa",
    "psa",
    "solubility",
    "clearance",
    "permeability",
    "lipinski",
    "qed",
}

_NUMERIC_VALUE_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?$")
_NUMERIC_RANGE_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*-\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?$"
)
_FRACTION_SCORE_PATTERN = re.compile(r"^\d+\s*/\s*\d+$")
logger = logging.getLogger(__name__)


def reject_reason_for_non_smiles_text(smiles: Any) -> str:
    """Return a cheap pre-screen reason for obvious non-SMILES text."""
    text = str(smiles or "").strip()
    if not text:
        return "empty SMILES"
    if set(text) <= {"-", "_", "="}:
        return "markdown/table separator is not a SMILES string"
    if _NUMERIC_RANGE_PATTERN.match(text):
        return "numeric range is not a SMILES string"
    if _FRACTION_SCORE_PATTERN.match(text):
        return "fraction score is not a SMILES string"
    if _NUMERIC_VALUE_PATTERN.match(text):
        return "numeric value is not a SMILES string"
    lowered = text.casefold()
    if lowered in NON_SMILES_PLACEHOLDERS:
        return "placeholder is not a SMILES string"
    if lowered in NON_SMILES_PROPERTY_NAMES:
        return "property name is not a SMILES string"
    if "\n" in text or "\r" in text:
        return "multi-line text is not a SMILES string"
    if text.startswith("```"):
        return "markdown code fence is not a SMILES string"
    if set(text) == {"#"}:
        return "markdown heading marker is not a SMILES string"
    if text.startswith("####"):
        return "markdown heading text is not a SMILES string"
    return ""


def _rdkit_error_capture() -> Any:
    if rdBase is None or not hasattr(rdBase, "CaptureErrorLog"):
        return None
    try:
        return rdBase.CaptureErrorLog()
    except Exception:
        return None


def _captured_rdkit_error(capture: Any) -> str:
    if capture is None:
        return ""
    messages = str(getattr(capture, "messages", "") or "").strip()
    if not messages:
        return ""
    return " ".join(messages.split())


def parse_smiles_quiet(smiles: Any) -> tuple[Any, str, str]:
    """Parse a SMILES string without letting RDKit emit parse noise to stderr.

    Returns:
        (mol, canonical_smiles, error_message)
    """
    text = str(smiles or "").strip()
    prescreen_error = reject_reason_for_non_smiles_text(text)
    if prescreen_error:
        return None, "", prescreen_error
    if Chem is None:
        return None, text, ""
    capture = _rdkit_error_capture()
    try:
        mol = Chem.MolFromSmiles(text)
    except Exception as exc:
        captured = _captured_rdkit_error(capture)
        if captured:
            return None, "", f"RDKit parse error: {exc}; {captured}"
        return None, "", f"RDKit parse error: {exc}"
    if mol is None:
        captured = _captured_rdkit_error(capture)
        if captured:
            return None, "", f"RDKit could not parse SMILES: {captured}"
        return None, "", "RDKit could not parse SMILES"
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        canonical = text
    return mol, canonical, ""


def validate_smiles(smiles: Any) -> tuple[bool, str, str]:
    """Return (is_valid, canonical_smiles, error_message)."""
    mol, canonical, error = parse_smiles_quiet(smiles)
    if error:
        return False, "", error
    if Chem is None:
        return True, canonical, ""
    if mol is None:
        return False, "", "RDKit could not parse SMILES"
    return True, canonical, ""


def is_valid_smiles(smiles: Any) -> bool:
    """Return True when the value is a chemically valid SMILES string."""
    if smiles is None or pd.isna(smiles) or smiles.strip() == '':
        return False
    ok, _canonical, _error = validate_smiles(smiles)
    return ok


def validate_smiles_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    verbose: bool = True,
) -> pd.DataFrame:
    """Return a copy of the DataFrame with invalid SMILES rows removed."""
    if smiles_col not in df.columns:
        raise ValueError(f"DataFrame must contain a {smiles_col!r} column")

    original_count = len(df)
    valid_mask = df[smiles_col].apply(is_valid_smiles)
    cleaned_df = df[valid_mask].reset_index(drop=True)

    if verbose:
        num_valid = len(cleaned_df)
        num_invalid = original_count - num_valid
        success_rate = (num_valid / original_count * 100) if original_count else 0.0
        logger.info("Original entries: %s", original_count)
        logger.info("Valid SMILES: %s", num_valid)
        logger.info("Invalid SMILES: %s", num_invalid)
        logger.info("Success rate: %.1f%%", success_rate)

    return cleaned_df


def validate_smiles_batch_messages(smiles_list: Any) -> dict[str, str]:
    """Return {input_smiles: "valid" or a validation error message}."""
    if not isinstance(smiles_list, list):
        return {"": "smiles_list must be a list of SMILES strings"}

    results: dict[str, str] = {}
    for item in smiles_list:
        smiles = str(item or "").strip()
        ok, _canonical, error = validate_smiles(smiles)
        results[smiles] = "valid" if ok else error or "invalid SMILES"
    return results


def split_valid_molecules(
    molecules: list[dict[str, Any]] | Any,
    *,
    smiles_keys: tuple[str, ...] = ("smiles", "SMILES"),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split molecule dictionaries into RDKit-valid and invalid lists."""
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    if not isinstance(molecules, list):
        return valid, invalid

    seen_canonical: set[str] = set()
    for item in molecules:
        if not isinstance(item, dict):
            invalid.append({"smiles": str(item), "error": "molecule entry is not an object"})
            continue
        smiles = next((item.get(key) for key in smiles_keys if item.get(key)), "")
        ok, canonical, error = validate_smiles(smiles)
        if not ok:
            invalid.append(
                {
                    "smiles": str(smiles or ""),
                    "error": error,
                    "agent_type": item.get("agent_type", ""),
                    "action": item.get("action") or item.get("modification_type") or "",
                }
            )
            continue
        dedupe_key = canonical or str(smiles)
        if dedupe_key in seen_canonical:
            invalid.append(
                {
                    "smiles": str(smiles or ""),
                    "canonical_smiles": canonical,
                    "error": "duplicate canonical SMILES",
                    "agent_type": item.get("agent_type", ""),
                    "action": item.get("action") or item.get("modification_type") or "",
                }
            )
            continue
        seen_canonical.add(dedupe_key)
        next_item = dict(item)
        if "smiles" in next_item:
            next_item["smiles"] = str(smiles)
        if "SMILES" in next_item:
            next_item["SMILES"] = str(smiles)
        next_item["canonical_smiles"] = canonical
        valid.append(next_item)
    return valid, invalid


def compact_invalid_molecules(items: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    """Return a small frontend-safe preview of invalid molecules."""
    preview: list[dict[str, Any]] = []
    for item in items[:limit]:
        preview.append(
            {
                "smiles": str(item.get("smiles") or "")[:160],
                "canonical_smiles": str(item.get("canonical_smiles") or "")[:160],
                "error": str(item.get("error") or "")[:240],
                "agent_type": str(item.get("agent_type") or "")[:80],
                "action": str(item.get("action") or "")[:160],
            }
        )
    return preview


def tanimoto_similarity_to_root(
    smiles: Any,
    root_smiles: Any,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> float | None:
    """Return Morgan-fingerprint Tanimoto similarity to the root molecule."""
    if Chem is None or DataStructs is None or AllChem is None:
        return None
    smiles_text = str(smiles or "").strip()
    root_text = str(root_smiles or "").strip()
    if not smiles_text or not root_text:
        return None
    try:
        mol, _canonical, mol_error = parse_smiles_quiet(smiles_text)
        root_mol, _root_canonical, root_error = parse_smiles_quiet(root_text)
        if mol is None or root_mol is None or mol_error or root_error:
            return None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        root_fingerprint = AllChem.GetMorganFingerprintAsBitVect(root_mol, radius, nBits=n_bits)
        return round(float(DataStructs.TanimotoSimilarity(fingerprint, root_fingerprint)), 4)
    except Exception:
        return None
