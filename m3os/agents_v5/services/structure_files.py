"""Utilities for uploaded molecular structure files.

This module handles machine-readable ligand and protein uploads before the
agent session starts. SDF files are converted to canonical SMILES strings, and
PDB files are converted to protein FASTA sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from rdkit import Chem


SUPPORTED_STRUCTURE_SUFFIXES = {".sdf", ".sd", ".pdb", ".ent"}
SDF_SUFFIXES = {".sdf", ".sd"}
PDB_SUFFIXES = {".pdb", ".ent"}

_MODIFIED_RESIDUE_MAP = {
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}


@dataclass(frozen=True)
class PDBSequenceRecord:
    """Protein sequence extracted from a PDB chain."""

    pdb_id: str
    chain_id: str
    sequence: str
    source: str


@dataclass(frozen=True)
class StructureFileResult:
    """Processed representation of one uploaded SDF or PDB file."""

    path: str
    source_type: str
    smiles: List[str]
    protein_sequences: List[PDBSequenceRecord]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def read_sdf(
    path: str | Path,
    *,
    sanitize: bool = True,
    remove_hs: bool = False,
) -> List[Chem.Mol]:
    """Read valid molecules from an SDF file with RDKit."""
    source = _validated_file(path)
    if source.suffix.lower() not in SDF_SUFFIXES:
        raise ValueError(f"Expected an SDF file, got: {source}")

    supplier = Chem.SDMolSupplier(str(source), sanitize=sanitize, removeHs=remove_hs)
    molecules = [mol for mol in supplier if mol is not None]
    if not molecules:
        raise ValueError(f"No valid molecules could be read from SDF file: {source}")
    return molecules


def read_pdb(path: str | Path) -> str:
    """Read raw PDB text from disk."""
    source = _validated_file(path)
    if source.suffix.lower() not in PDB_SUFFIXES:
        raise ValueError(f"Expected a PDB file, got: {source}")
    return source.read_text(encoding="utf-8", errors="replace")


def sdf_file_to_smiles(
    path: str | Path,
    *,
    canonical: bool = True,
    isomeric: bool = True,
    deduplicate: bool = True,
) -> List[str]:
    """Convert all valid molecules in an SDF file to SMILES strings."""
    smiles_values: List[str] = []
    seen: set[str] = set()
    for molecule in read_sdf(path):
        smiles = Chem.MolToSmiles(
            molecule,
            canonical=canonical,
            isomericSmiles=isomeric,
        ).strip()
        if not smiles:
            continue
        if deduplicate and smiles in seen:
            continue
        smiles_values.append(smiles)
        seen.add(smiles)

    if not smiles_values:
        raise ValueError(f"No valid SMILES could be generated from SDF file: {path}")
    return smiles_values


def pdb_file_to_protein_sequence(
    path: str | Path,
    *,
    chain_id: Optional[str] = None,
    prefer_seqres: bool = True,
) -> str:
    """Convert a PDB file to protein sequence FASTA text."""
    records = extract_pdb_sequence_records(
        path,
        chain_id=chain_id,
        prefer_seqres=prefer_seqres,
    )
    if not records:
        raise ValueError(f"No protein sequence could be extracted from PDB file: {path}")
    return format_pdb_sequences_as_fasta(records)


def extract_pdb_sequence_records(
    path: str | Path,
    *,
    chain_id: Optional[str] = None,
    prefer_seqres: bool = True,
) -> List[PDBSequenceRecord]:
    """Extract protein sequences from PDB SEQRES records or ATOM residues."""
    source = _validated_file(path)
    if source.suffix.lower() not in PDB_SUFFIXES:
        raise ValueError(f"Expected a PDB file, got: {source}")

    records: List[PDBSequenceRecord] = []
    if prefer_seqres:
        records = _extract_seqres_records(source, chain_id=chain_id)
    if not records:
        records = _extract_atom_records(source, chain_id=chain_id)

    if chain_id is not None and not records:
        raise ValueError(f"Chain '{chain_id}' was not found in PDB file: {source}")
    return records


def collect_structure_files(
    paths: Optional[Sequence[str | Path] | str | Path] = None,
    directory: Optional[str | Path] = None,
) -> List[str]:
    """Collect supported SDF/PDB paths from explicit paths and one directory."""
    candidate_paths = _normalize_path_sequence(paths)

    if directory:
        root = Path(directory).expanduser()
        if root.exists() and root.is_dir():
            candidate_paths.extend(
                child
                for child in sorted(root.iterdir())
                if child.is_file() and child.suffix.lower() in SUPPORTED_STRUCTURE_SUFFIXES
            )

    normalized: List[str] = []
    seen: set[str] = set()
    for path in candidate_paths:
        expanded = Path(path).expanduser()
        if not expanded.is_file() or expanded.suffix.lower() not in SUPPORTED_STRUCTURE_SUFFIXES:
            continue
        resolved = str(expanded.resolve())
        if resolved not in seen:
            normalized.append(resolved)
            seen.add(resolved)
    return normalized


def process_uploaded_structure_files(
    paths: Optional[Sequence[str | Path] | str | Path] = None,
    *,
    directory: Optional[str | Path] = None,
) -> List[StructureFileResult]:
    """Process uploaded SDF/PDB files and return conversion results."""
    results: List[StructureFileResult] = []
    for file_path in collect_structure_files(paths, directory):
        path = Path(file_path)
        suffix = path.suffix.lower()
        try:
            if suffix in SDF_SUFFIXES:
                results.append(
                    StructureFileResult(
                        path=str(path),
                        source_type="SDF",
                        smiles=sdf_file_to_smiles(path),
                        protein_sequences=[],
                    )
                )
            elif suffix in PDB_SUFFIXES:
                protein_sequences = extract_pdb_sequence_records(path)
                if not protein_sequences:
                    raise ValueError(f"No protein sequence could be extracted from PDB file: {path}")
                results.append(
                    StructureFileResult(
                        path=str(path),
                        source_type="PDB",
                        smiles=[],
                        protein_sequences=protein_sequences,
                    )
                )
        except Exception as exc:
            results.append(
                StructureFileResult(
                    path=str(path),
                    source_type="SDF" if suffix in SDF_SUFFIXES else "PDB",
                    smiles=[],
                    protein_sequences=[],
                    error=str(exc),
                )
            )
    return results


def build_structure_file_context(
    paths: Optional[Sequence[str | Path] | str | Path] = None,
    *,
    directory: Optional[str | Path] = None,
    max_smiles: int = 50,
    protein_fasta_path: Optional[str | Path] = None,
) -> Optional[str]:
    """Build a system-prompt block from uploaded SDF/PDB conversions."""
    results = process_uploaded_structure_files(paths, directory=directory)
    if not results:
        return None

    successful = [result for result in results if result.ok]
    failures = [result for result in results if not result.ok]
    if not successful and not failures:
        return None

    lines: List[str] = [
        "[USER-UPLOADED STRUCTURE FILE CONTEXT]",
        "The following SDF/PDB files were processed before this agent session started.",
        "Use uploaded SDF SMILES as machine-ready ligand input and uploaded PDB FASTA as target-protein context unless the user explicitly overrides them.",
    ]
    authoritative_fasta = None
    if protein_fasta_path:
        authoritative_fasta = str(Path(protein_fasta_path).expanduser().resolve())
        lines.append(f"Authoritative protein FASTA path: {authoritative_fasta}")

    for index, result in enumerate(results, start=1):
        path = Path(result.path)
        lines.extend(
            [
                "",
                f"## Structure File {index}: {path.name}",
                f"Path: {result.path}",
                f"Type: {result.source_type}",
            ]
        )
        if result.error:
            lines.append(f"Processing error: {result.error}")
            continue
        if result.source_type == "SDF":
            lines.append(f"Molecule count: {len(result.smiles)}")
            if result.smiles:
                lines.append(f"Primary SMILES: {result.smiles[0]}")
                lines.append("SMILES list:")
                for smiles_index, smiles in enumerate(result.smiles[:max_smiles], start=1):
                    lines.append(f"{smiles_index}. {smiles}")
                if len(result.smiles) > max_smiles:
                    lines.append(f"... {len(result.smiles) - max_smiles} additional SMILES omitted.")
        elif result.source_type == "PDB":
            lines.append(f"Protein chain count: {len(result.protein_sequences)}")
            if authoritative_fasta:
                lines.append(
                    "Protein sequence omitted from this prompt; protein-aware tools must "
                    "read the authoritative FASTA path above."
                )
            else:
                lines.append("Protein sequence FASTA:")
                lines.append(format_pdb_sequences_as_fasta(result.protein_sequences))

    lines.append("[END USER-UPLOADED STRUCTURE FILE CONTEXT]")
    return "\n".join(lines)


def format_pdb_sequences_as_fasta(records: Sequence[PDBSequenceRecord]) -> str:
    """Format extracted PDB chain sequences as FASTA text."""
    fasta_lines: List[str] = []
    for record in records:
        chain_label = record.chain_id or "_"
        fasta_lines.append(f">{record.pdb_id}|chain {chain_label}|source={record.source}")
        fasta_lines.extend(_wrap_sequence(record.sequence))
    return "\n".join(fasta_lines)


def _validated_file(path: str | Path) -> Path:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not source.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return source


def _normalize_path_sequence(
    paths: Optional[Sequence[str | Path] | str | Path],
) -> List[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(item) for item in paths]


def _extract_seqres_records(
    source: Path,
    *,
    chain_id: Optional[str] = None,
) -> List[PDBSequenceRecord]:
    chains: dict[str, List[str]] = {}
    for line in read_pdb(source).splitlines():
        if not line.startswith("SEQRES"):
            continue
        current_chain = line[11].strip() or "_"
        if chain_id is not None and current_chain != chain_id:
            continue
        chains.setdefault(current_chain, []).extend(line[19:70].split())

    records: List[PDBSequenceRecord] = []
    for current_chain, residue_names in sorted(chains.items()):
        sequence = _residue_names_to_sequence(residue_names)
        if _looks_like_protein_sequence(sequence):
            records.append(
                PDBSequenceRecord(
                    pdb_id=source.stem,
                    chain_id=current_chain,
                    sequence=sequence,
                    source="SEQRES",
                )
            )
    return records


def _extract_atom_records(
    source: Path,
    *,
    chain_id: Optional[str] = None,
) -> List[PDBSequenceRecord]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(source.stem, str(source))
    try:
        model = next(structure.get_models())
    except StopIteration:
        return []

    records: List[PDBSequenceRecord] = []
    for chain in model:
        current_chain = str(chain.id).strip() or "_"
        if chain_id is not None and current_chain != chain_id:
            continue
        residue_names: List[str] = []
        seen_residues: set[tuple[str, int, str]] = set()
        for residue in chain:
            residue_id = residue.id
            if residue_id[0].strip():
                continue
            dedupe_key = (str(residue_id[0]), int(residue_id[1]), str(residue_id[2]).strip())
            if dedupe_key in seen_residues:
                continue
            residue_names.append(str(residue.get_resname()).strip())
            seen_residues.add(dedupe_key)
        sequence = _residue_names_to_sequence(residue_names)
        if _looks_like_protein_sequence(sequence):
            records.append(
                PDBSequenceRecord(
                    pdb_id=source.stem,
                    chain_id=current_chain,
                    sequence=sequence,
                    source="ATOM",
                )
            )
    return records


def _residue_names_to_sequence(residue_names: Iterable[str]) -> str:
    letters: List[str] = []
    for residue_name in residue_names:
        letters.append(_residue_name_to_one_letter(residue_name))
    return "".join(letters)


def _looks_like_protein_sequence(sequence: str) -> bool:
    return bool(sequence) and any(letter != "X" for letter in sequence)


def _residue_name_to_one_letter(residue_name: str) -> str:
    normalized = residue_name.strip().upper()
    if not normalized:
        return "X"
    if normalized in _MODIFIED_RESIDUE_MAP:
        return _MODIFIED_RESIDUE_MAP[normalized]
    try:
        return str(seq1(normalized, custom_map=_MODIFIED_RESIDUE_MAP, undef_code="X"))
    except Exception:
        return "X"


def _wrap_sequence(sequence: str, width: int = 80) -> List[str]:
    if not sequence:
        return [""]
    return [sequence[index : index + width] for index in range(0, len(sequence), width)]
