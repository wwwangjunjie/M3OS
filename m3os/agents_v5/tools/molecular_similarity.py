"""RDKit molecule/scaffold similarity utilities."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


FINGERPRINT_RADIUS = 2
FINGERPRINT_SIZE = 2048
FINGERPRINT_METHOD = "rdkit_morgan_tanimoto"
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=FINGERPRINT_RADIUS,
    fpSize=FINGERPRINT_SIZE,
)


@dataclass(frozen=True)
class SimilarityHit:
    """A scored candidate molecule or scaffold."""

    smiles: str
    similarity: float
    index: int


def _fingerprint_from_smiles(smiles: str):
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None
    return _MORGAN_GENERATOR.GetFingerprint(mol)


def calculate_tanimoto_similarity(smiles_a: str, smiles_b: str) -> float | None:
    """Return Morgan-fingerprint Tanimoto similarity for two SMILES strings."""
    fp_a = _fingerprint_from_smiles(smiles_a)
    fp_b = _fingerprint_from_smiles(smiles_b)
    if fp_a is None or fp_b is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _score_candidate(query_fp, item: tuple[int, str]) -> SimilarityHit | None:
    index, smiles = item
    candidate_fp = _fingerprint_from_smiles(smiles)
    if candidate_fp is None:
        return None
    return SimilarityHit(
        smiles=smiles,
        similarity=float(DataStructs.TanimotoSimilarity(query_fp, candidate_fp)),
        index=index,
    )


def batch_tanimoto_similarity_search(
    query_smiles: str,
    candidate_smiles: Sequence[str] | Iterable[str],
    *,
    max_workers: int | None = None,
) -> list[SimilarityHit]:
    """Score candidates against a query SMILES/scaffold using worker threads.

    Invalid candidate SMILES are skipped. Results are sorted by similarity
    descending and then by original candidate order for deterministic ties.
    """
    query_fp = _fingerprint_from_smiles(query_smiles)
    if query_fp is None:
        raise ValueError(f"Invalid query SMILES: {query_smiles}")

    candidates = list(candidate_smiles)
    if not candidates:
        return []
    workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
    workers = max(1, int(workers))

    if workers == 1 or len(candidates) == 1:
        hits = [_score_candidate(query_fp, item) for item in enumerate(candidates)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            hits = list(
                executor.map(
                    lambda item: _score_candidate(query_fp, item),
                    enumerate(candidates),
                )
            )

    valid_hits = [hit for hit in hits if hit is not None]
    valid_hits.sort(key=lambda hit: (-hit.similarity, hit.index, hit.smiles))
    return valid_hits


def molecular_similarity_search(smiles_a: str, smiles_b: str) -> str:
    """Calculate structural similarity between two molecules or scaffolds.

    Args:
        smiles_a: First molecule/scaffold SMILES.
        smiles_b: Second molecule/scaffold SMILES.

    Returns:
        JSON string with Morgan-fingerprint Tanimoto similarity, or an error
        string for invalid input matching the existing MCP tool style.
    """
    similarity = calculate_tanimoto_similarity(smiles_a, smiles_b)
    if similarity is None:
        if _fingerprint_from_smiles(smiles_a) is None:
            return f"Error: Invalid SMILES for molecule A: {smiles_a}"
        return f"Error: Invalid SMILES for molecule B: {smiles_b}"

    result = {
        "input_molecules": {
            "molecule_a": smiles_a,
            "molecule_b": smiles_b,
        },
        "fingerprint_method": FINGERPRINT_METHOD,
        "fingerprint_radius": FINGERPRINT_RADIUS,
        "fingerprint_size": FINGERPRINT_SIZE,
        "similarity_metric": "tanimoto",
        "similarity": similarity,
    }
    return json.dumps(result, indent=2, ensure_ascii=False)
