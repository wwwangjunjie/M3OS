"""RDKit molecule image helpers for MCP tools.

The functions in this module save PNG files and return filesystem paths so
vision models can consume the same two-dimensional topology diagrams a
medicinal chemist would inspect.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Draw, rdFMCS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE_SUBDIR = "molecule_images"

COMMON_COLOR = (0.38, 0.70, 0.42)
CHANGE_COLOR = (0.93, 0.48, 0.18)
STEREO_CHANGE_COLOR = (0.58, 0.22, 0.86)

DEFAULT_TOPOLOGY_SIZE = (520, 420)
DEFAULT_SINGLE_DIFF_SIZE = (620, 460)
DEFAULT_PAIR_DIFF_SIZE = (1240, 460)
DEFAULT_MCS_TIMEOUT_SECONDS = 20


def generate_molecule_topology_image(
    smiles: str,
    output_dir: Optional[str] = None,
    image_prefix: str = "molecule_topology",
    width: int = DEFAULT_TOPOLOGY_SIZE[0],
    height: int = DEFAULT_TOPOLOGY_SIZE[1],
    show_atom_indices: bool = False,
) -> Dict[str, Any]:
    """Generate a single-molecule 2D topology PNG and return its path."""
    mol, canonical_smiles = _parse_smiles(smiles, "smiles")
    _prepare_molecule_for_drawing(mol, show_atom_indices=show_atom_indices)

    size = _normal_size(width, height, DEFAULT_TOPOLOGY_SIZE)
    image = Draw.MolToImage(mol, size=size)
    output_path = _output_path(
        output_dir=output_dir,
        image_prefix=image_prefix,
        content_key=f"topology|{canonical_smiles}|{size}|{show_atom_indices}",
    )
    _save_png(image, output_path)
    return _result_payload(
        image_path=output_path,
        image_type="topology",
        canonical_smiles=canonical_smiles,
        width=size[0],
        height=size[1],
        method="rdkit_2d",
    )


def generate_molecule_difference_image(
    smiles: str,
    reference_smiles: str,
    output_dir: Optional[str] = None,
    image_prefix: str = "molecule_difference",
    width: int = DEFAULT_SINGLE_DIFF_SIZE[0],
    height: int = DEFAULT_SINGLE_DIFF_SIZE[1],
    mcs_timeout: int = DEFAULT_MCS_TIMEOUT_SECONDS,
    show_atom_indices: bool = False,
) -> Dict[str, Any]:
    """Draw one molecule and highlight its differences from a reference."""
    reference_mol, reference_canonical = _parse_smiles(reference_smiles, "reference_smiles")
    target_mol, target_canonical = _parse_smiles(smiles, "smiles")
    _prepare_molecule_for_drawing(reference_mol, show_atom_indices=False)
    _prepare_molecule_for_drawing(target_mol, show_atom_indices=show_atom_indices)

    size = _normal_size(width, height, DEFAULT_SINGLE_DIFF_SIZE)
    details = _mcs_details(
        reference_mol,
        target_mol,
        mcs_timeout=mcs_timeout,
    )
    if details["reference_match"] and details["target_match"]:
        _compute_aligned_2d_coords(
            reference_mol,
            target_mol,
            details["reference_match"],
            details["target_match"],
        )
        (
            _reference_stereo_atoms,
            target_stereo_atoms,
            _reference_stereo_bonds,
            target_stereo_bonds,
        ) = _stereo_difference_details(
            reference_mol,
            target_mol,
            details["reference_match"],
            details["target_match"],
        )
        target_atoms, target_bonds, target_atom_colors, target_bond_colors, changed_atoms = (
            _highlight_details(
                target_mol,
                details["target_match"],
                details["target_common_bonds"],
                stereo_atoms=target_stereo_atoms,
                stereo_bonds=target_stereo_bonds,
                include_common=False,
            )
        )
        image = Draw.MolToImage(
            target_mol,
            size=size,
            highlightAtoms=target_atoms,
            highlightBonds=target_bonds,
            highlightAtomColors=target_atom_colors,
            highlightBondColors=target_bond_colors,
        )
        image = _add_caption(
            image,
            "Different vs reference: orange=target-only/changed | purple=stereo",
        )
        method = "rdkit_mcs_single"
    else:
        AllChem.Compute2DCoords(target_mol)
        target_atoms = [atom.GetIdx() for atom in target_mol.GetAtoms()]
        target_bonds = [bond.GetIdx() for bond in target_mol.GetBonds()]
        target_atom_colors = {atom_idx: CHANGE_COLOR for atom_idx in target_atoms}
        target_bond_colors = {bond_idx: CHANGE_COLOR for bond_idx in target_bonds}
        image = Draw.MolToImage(
            target_mol,
            size=size,
            highlightAtoms=target_atoms,
            highlightBonds=target_bonds,
            highlightAtomColors=target_atom_colors,
            highlightBondColors=target_bond_colors,
        )
        image = _add_caption(image, "No stable MCS found; all target atoms/bonds highlighted")
        target_stereo_atoms = []
        target_stereo_bonds = []
        changed_atoms = list(range(target_mol.GetNumAtoms()))
        method = "rdkit_no_mcs_single"

    output_path = _output_path(
        output_dir=output_dir,
        image_prefix=image_prefix,
        content_key=(
            "single_diff|"
            f"{reference_canonical}|{target_canonical}|{size}|{mcs_timeout}|{show_atom_indices}"
        ),
    )
    _save_png(image, output_path)
    return _result_payload(
        image_path=output_path,
        image_type="single_difference",
        canonical_smiles=target_canonical,
        reference_canonical_smiles=reference_canonical,
        width=size[0],
        height=size[1],
        method=method,
        mcs_atoms=len(details["target_match"]),
        mcs_bonds=len(details["target_common_bonds"]),
        changed_atoms=len(changed_atoms),
        highlighted_atoms=len(target_atoms),
        highlighted_bonds=len(target_bonds),
        stereo_changed_atoms=len(target_stereo_atoms),
        stereo_changed_bonds=len(target_stereo_bonds),
        mcs_canceled=bool(details["mcs_canceled"]),
    )


def generate_molecule_pair_difference_image(
    before_smiles: str,
    after_smiles: str,
    output_dir: Optional[str] = None,
    image_prefix: str = "molecule_pair_difference",
    width: int = DEFAULT_PAIR_DIFF_SIZE[0],
    height: int = DEFAULT_PAIR_DIFF_SIZE[1],
    mcs_timeout: int = DEFAULT_MCS_TIMEOUT_SECONDS,
    show_atom_indices: bool = False,
) -> Dict[str, Any]:
    """Draw before/after molecules side by side with MCS differences highlighted."""
    before_mol, before_canonical = _parse_smiles(before_smiles, "before_smiles")
    after_mol, after_canonical = _parse_smiles(after_smiles, "after_smiles")
    _prepare_molecule_for_drawing(before_mol, show_atom_indices=show_atom_indices)
    _prepare_molecule_for_drawing(after_mol, show_atom_indices=show_atom_indices)

    size = _normal_size(width, height, DEFAULT_PAIR_DIFF_SIZE)
    sub_size = (max(260, size[0] // 2), max(260, size[1]))
    details = _mcs_details(before_mol, after_mol, mcs_timeout=mcs_timeout)

    if details["reference_match"] and details["target_match"]:
        _compute_aligned_2d_coords(
            before_mol,
            after_mol,
            details["reference_match"],
            details["target_match"],
        )
        (
            before_stereo_atoms,
            after_stereo_atoms,
            before_stereo_bonds,
            after_stereo_bonds,
        ) = _stereo_difference_details(
            before_mol,
            after_mol,
            details["reference_match"],
            details["target_match"],
        )
        before_atoms, before_bonds, before_atom_colors, before_bond_colors, before_changed_atoms = (
            _highlight_details(
                before_mol,
                details["reference_match"],
                details["reference_common_bonds"],
                stereo_atoms=before_stereo_atoms,
                stereo_bonds=before_stereo_bonds,
            )
        )
        after_atoms, after_bonds, after_atom_colors, after_bond_colors, after_changed_atoms = (
            _highlight_details(
                after_mol,
                details["target_match"],
                details["target_common_bonds"],
                stereo_atoms=after_stereo_atoms,
                stereo_bonds=after_stereo_bonds,
            )
        )
        image = Draw.MolsToGridImage(
            [before_mol, after_mol],
            molsPerRow=2,
            subImgSize=sub_size,
            legends=["Before", "After"],
            highlightAtomLists=[before_atoms, after_atoms],
            highlightBondLists=[before_bonds, after_bonds],
            highlightAtomColors=[before_atom_colors, after_atom_colors],
            highlightBondColors=[before_bond_colors, after_bond_colors],
            useSVG=False,
        )
        image = _add_caption(
            image,
            "Pair difference: green=shared MCS | orange=changed/added/removed | purple=stereo",
        )
        method = "rdkit_mcs_pair"
    else:
        for mol in (before_mol, after_mol):
            AllChem.Compute2DCoords(mol)
        image = Draw.MolsToGridImage(
            [before_mol, after_mol],
            molsPerRow=2,
            subImgSize=sub_size,
            legends=["Before", "After"],
            useSVG=False,
        )
        image = _add_caption(image, "No stable MCS found; showing before/after topology")
        before_stereo_atoms = []
        after_stereo_atoms = []
        before_stereo_bonds = []
        after_stereo_bonds = []
        before_changed_atoms = list(range(before_mol.GetNumAtoms()))
        after_changed_atoms = list(range(after_mol.GetNumAtoms()))
        method = "rdkit_plain_pair"

    output_path = _output_path(
        output_dir=output_dir,
        image_prefix=image_prefix,
        content_key=(
            "pair_diff|"
            f"{before_canonical}|{after_canonical}|{size}|{mcs_timeout}|{show_atom_indices}"
        ),
    )
    _save_png(image, output_path)
    return _result_payload(
        image_path=output_path,
        image_type="pair_difference",
        canonical_smiles=after_canonical,
        before_canonical_smiles=before_canonical,
        after_canonical_smiles=after_canonical,
        width=image.width,
        height=image.height,
        method=method,
        mcs_atoms=len(details["target_match"]),
        mcs_bonds=len(details["target_common_bonds"]),
        before_changed_atoms=len(before_changed_atoms),
        after_changed_atoms=len(after_changed_atoms),
        before_stereo_changed_atoms=len(before_stereo_atoms),
        after_stereo_changed_atoms=len(after_stereo_atoms),
        before_stereo_changed_bonds=len(before_stereo_bonds),
        after_stereo_changed_bonds=len(after_stereo_bonds),
        mcs_canceled=bool(details["mcs_canceled"]),
    )


def _parse_smiles(smiles: str, field_name: str) -> Tuple[Chem.Mol, str]:
    text = str(smiles or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required.")
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"RDKit could not parse {field_name}: {text}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    return Chem.Mol(mol), canonical


def _prepare_molecule_for_drawing(mol: Chem.Mol, *, show_atom_indices: bool) -> None:
    AllChem.Compute2DCoords(mol)
    if show_atom_indices:
        for atom in mol.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()))


def _mcs_details(
    reference_mol: Chem.Mol,
    target_mol: Chem.Mol,
    *,
    mcs_timeout: int,
) -> Dict[str, Any]:
    timeout = max(1, int(mcs_timeout or DEFAULT_MCS_TIMEOUT_SECONDS))
    try:
        mcs = rdFMCS.FindMCS(
            [reference_mol, target_mol],
            timeout=timeout,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchValences=True,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
    except Exception:
        return {
            "reference_match": (),
            "target_match": (),
            "reference_common_bonds": [],
            "target_common_bonds": [],
            "mcs_canceled": False,
        }

    if not getattr(mcs, "smartsString", ""):
        return {
            "reference_match": (),
            "target_match": (),
            "reference_common_bonds": [],
            "target_common_bonds": [],
            "mcs_canceled": bool(getattr(mcs, "canceled", False)),
        }

    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    if mcs_mol is None:
        return {
            "reference_match": (),
            "target_match": (),
            "reference_common_bonds": [],
            "target_common_bonds": [],
            "mcs_canceled": bool(getattr(mcs, "canceled", False)),
        }

    reference_match = tuple(reference_mol.GetSubstructMatch(mcs_mol))
    target_match = tuple(target_mol.GetSubstructMatch(mcs_mol))
    if not reference_match or not target_match:
        return {
            "reference_match": (),
            "target_match": (),
            "reference_common_bonds": [],
            "target_common_bonds": [],
            "mcs_canceled": bool(getattr(mcs, "canceled", False)),
        }

    return {
        "reference_match": reference_match,
        "target_match": target_match,
        "reference_common_bonds": _common_bond_indices(reference_mol, mcs_mol, reference_match),
        "target_common_bonds": _common_bond_indices(target_mol, mcs_mol, target_match),
        "mcs_canceled": bool(getattr(mcs, "canceled", False)),
    }


def _common_bond_indices(
    mol: Chem.Mol,
    mcs_mol: Chem.Mol,
    mol_match: Sequence[int],
) -> List[int]:
    bond_indices = set()
    for mcs_bond in mcs_mol.GetBonds():
        begin = mol_match[mcs_bond.GetBeginAtomIdx()]
        end = mol_match[mcs_bond.GetEndAtomIdx()]
        mol_bond = mol.GetBondBetweenAtoms(begin, end)
        if mol_bond is not None:
            bond_indices.add(mol_bond.GetIdx())
    return sorted(bond_indices)


def _compute_aligned_2d_coords(
    reference_mol: Chem.Mol,
    target_mol: Chem.Mol,
    reference_match: Sequence[int],
    target_match: Sequence[int],
) -> None:
    AllChem.Compute2DCoords(reference_mol)
    rdBase.DisableLog("rdApp.error")
    try:
        AllChem.GenerateDepictionMatching2DStructure(
            target_mol,
            reference_mol,
            atomMap=list(zip(target_match, reference_match)),
        )
    except Exception:
        AllChem.Compute2DCoords(target_mol)
    finally:
        rdBase.EnableLog("rdApp.error")


def _atom_stereo_descriptor(atom: Chem.Atom) -> Optional[str]:
    if atom.HasProp("_CIPCode"):
        return atom.GetProp("_CIPCode")
    chiral_tag = atom.GetChiralTag()
    if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
        return str(chiral_tag)
    return None


def _bond_stereo_descriptor(bond: Chem.Bond) -> Optional[str]:
    stereo = bond.GetStereo()
    if stereo in (Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY):
        return None
    return str(stereo)


def _stereo_difference_details(
    reference_mol: Chem.Mol,
    target_mol: Chem.Mol,
    reference_match: Sequence[int],
    target_match: Sequence[int],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    reference_to_target = dict(zip(reference_match, target_match))
    reference_stereo_atoms = set()
    target_stereo_atoms = set()
    reference_stereo_bonds = set()
    target_stereo_bonds = set()

    for reference_idx, target_idx in reference_to_target.items():
        reference_atom = reference_mol.GetAtomWithIdx(reference_idx)
        target_atom = target_mol.GetAtomWithIdx(target_idx)
        reference_stereo = _atom_stereo_descriptor(reference_atom)
        target_stereo = _atom_stereo_descriptor(target_atom)
        if reference_stereo == target_stereo or (reference_stereo is None and target_stereo is None):
            continue
        reference_stereo_atoms.add(reference_idx)
        target_stereo_atoms.add(target_idx)
        for bond in reference_atom.GetBonds():
            reference_stereo_bonds.add(bond.GetIdx())
        for bond in target_atom.GetBonds():
            target_stereo_bonds.add(bond.GetIdx())

    for reference_bond in reference_mol.GetBonds():
        reference_begin = reference_bond.GetBeginAtomIdx()
        reference_end = reference_bond.GetEndAtomIdx()
        if reference_begin not in reference_to_target or reference_end not in reference_to_target:
            continue

        target_bond = target_mol.GetBondBetweenAtoms(
            reference_to_target[reference_begin],
            reference_to_target[reference_end],
        )
        if target_bond is None:
            continue

        reference_stereo = _bond_stereo_descriptor(reference_bond)
        target_stereo = _bond_stereo_descriptor(target_bond)
        if reference_stereo == target_stereo or (reference_stereo is None and target_stereo is None):
            continue

        reference_stereo_bonds.add(reference_bond.GetIdx())
        target_stereo_bonds.add(target_bond.GetIdx())
        for atom_idx in (reference_begin, reference_end, *reference_bond.GetStereoAtoms()):
            reference_stereo_atoms.add(atom_idx)
        for atom_idx in (
            target_bond.GetBeginAtomIdx(),
            target_bond.GetEndAtomIdx(),
            *target_bond.GetStereoAtoms(),
        ):
            target_stereo_atoms.add(atom_idx)

    return (
        sorted(reference_stereo_atoms),
        sorted(target_stereo_atoms),
        sorted(reference_stereo_bonds),
        sorted(target_stereo_bonds),
    )


def _highlight_details(
    mol: Chem.Mol,
    common_atoms: Sequence[int],
    common_bonds: Sequence[int],
    *,
    stereo_atoms: Sequence[int] = (),
    stereo_bonds: Sequence[int] = (),
    include_common: bool = True,
) -> Tuple[
    List[int],
    List[int],
    Dict[int, Tuple[float, float, float]],
    Dict[int, Tuple[float, float, float]],
    List[int],
]:
    common_atom_set = set(common_atoms)
    common_bond_set = set(common_bonds)
    stereo_atom_set = set(stereo_atoms)
    stereo_bond_set = set(stereo_bonds)
    changed_atom_set = {atom.GetIdx() for atom in mol.GetAtoms()} - common_atom_set
    changed_bond_set = {bond.GetIdx() for bond in mol.GetBonds()} - common_bond_set

    if include_common:
        highlighted_atoms = common_atom_set | changed_atom_set | stereo_atom_set
        highlighted_bonds = common_bond_set | changed_bond_set | stereo_bond_set
    else:
        highlighted_atoms = changed_atom_set | stereo_atom_set
        highlighted_bonds = changed_bond_set | stereo_bond_set
    atom_colors = {
        atom_idx: (
            STEREO_CHANGE_COLOR
            if atom_idx in stereo_atom_set
            else CHANGE_COLOR
            if atom_idx in changed_atom_set
            else COMMON_COLOR
        )
        for atom_idx in highlighted_atoms
    }
    bond_colors = {
        bond_idx: (
            STEREO_CHANGE_COLOR
            if bond_idx in stereo_bond_set
            else CHANGE_COLOR
            if bond_idx in changed_bond_set
            else COMMON_COLOR
        )
        for bond_idx in highlighted_bonds
    }
    return (
        sorted(highlighted_atoms),
        sorted(highlighted_bonds),
        atom_colors,
        bond_colors,
        sorted(changed_atom_set),
    )


def _add_caption(image: Image.Image, caption: str) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    pad_top = 44
    canvas = Image.new("RGB", (image.width, image.height + pad_top), "white")
    canvas.paste(image, (0, pad_top))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 14), caption, fill=(30, 30, 30))
    return canvas


def _save_png(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode not in {"RGB", "RGBA"}:
        image = image.convert("RGB")
    image.save(output_path, format="PNG")


def _output_path(output_dir: Optional[str], image_prefix: str, content_key: str) -> Path:
    directory = _resolve_output_dir(output_dir)
    digest = hashlib.sha1(content_key.encode("utf-8")).hexdigest()[:12]
    prefix = _safe_filename_part(image_prefix or "molecule_image")
    return directory / f"{prefix}_{digest}.png"


def _resolve_output_dir(output_dir: Optional[str]) -> Path:
    text = str(output_dir or "").strip()
    if not text:
        return _current_tmp_dir() / DEFAULT_IMAGE_SUBDIR
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "tmp":
        return _current_tmp_dir().joinpath(*path.parts[1:])
    return PROJECT_ROOT / path


def _current_tmp_dir() -> Path:
    value = os.getenv("M3OS_TMP_DIR")
    if not value or not value.strip():
        return PROJECT_ROOT / "tmp"
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _safe_filename_part(value: str, max_length: int = 80) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._")
    return (safe or "molecule_image")[:max_length]


def _normal_size(width: int, height: int, default: Tuple[int, int]) -> Tuple[int, int]:
    try:
        w = int(width)
        h = int(height)
    except Exception:
        return default
    if w < 160 or h < 160:
        return default
    return min(w, 2400), min(h, 1600)


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _result_payload(image_path: Path, image_type: str, **metadata: Any) -> Dict[str, Any]:
    return {
        "valid": True,
        "image_type": image_type,
        "image_path": str(image_path.resolve()),
        "relative_image_path": _relative_path(image_path),
        **metadata,
    }
