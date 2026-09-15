"""Build aligned, parent-relative molecule SVG assets for an M3OS report.

Every non-root depiction highlights the atoms and bonds changed from its direct
parent.  The implementation combines M3OS's existing MCS difference drawing
with a permissive fallback for heteroatom/ring swaps and a deletion-boundary
fallback learned from the compound SAR report renderer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, rdDepictor, rdFMCS, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


DIFF_COLOR = (1.00, 0.64, 0.28, 0.38)
DIFF_BOND_COLOR = (1.00, 0.55, 0.18, 0.62)
STEREO_COLOR = (0.55, 0.28, 0.82, 0.46)
STEREO_BOND_COLOR = (0.48, 0.20, 0.76, 0.68)
SVG_SIZE = (760, 520)
TOPOLOGY_SVG_SIZE = (300, 200)


def _parse(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.Mol(mol)


def _find_mcs(parent: Chem.Mol, child: Chem.Mol, *, strict: bool) -> Any:
    if strict:
        return rdFMCS.FindMCS(
            [parent, child],
            timeout=20,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchValences=True,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
    return rdFMCS.FindMCS(
        [parent, child],
        timeout=20,
        atomCompare=rdFMCS.AtomCompare.CompareAny,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        matchValences=False,
    )


def _mcs_matches(parent: Chem.Mol, child: Chem.Mol) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    strict = _find_mcs(parent, child, strict=True)
    selected = strict
    method = "strict_mcs"
    minimum = min(parent.GetNumHeavyAtoms(), child.GetNumHeavyAtoms())
    if not strict.smartsString or strict.numAtoms < 0.70 * minimum:
        permissive = _find_mcs(parent, child, strict=False)
        if permissive.smartsString and permissive.numAtoms > strict.numAtoms:
            selected = permissive
            method = "permissive_mcs"
    pattern = Chem.MolFromSmarts(selected.smartsString) if selected.smartsString else None
    if pattern is None:
        return (), (), "no_mcs"
    return (
        tuple(parent.GetSubstructMatch(pattern)),
        tuple(child.GetSubstructMatch(pattern)),
        method,
    )


def _align(
    parent: Chem.Mol,
    child: Chem.Mol,
    parent_match: tuple[int, ...],
    child_match: tuple[int, ...],
) -> None:
    rdDepictor.Compute2DCoords(parent)
    if not parent_match or not child_match:
        rdDepictor.Compute2DCoords(child)
        return
    coordinates = {}
    conformer = parent.GetConformer()
    for parent_idx, child_idx in zip(parent_match, child_match):
        point = conformer.GetAtomPosition(parent_idx)
        coordinates[child_idx] = Chem.rdGeometry.Point2D(point.x, point.y)
    rdBase.DisableLog("rdApp.error")
    try:
        rdDepictor.Compute2DCoords(child, coordMap=coordinates)
    except Exception:
        rdDepictor.Compute2DCoords(child)
    finally:
        rdBase.EnableLog("rdApp.error")


def _atom_signature(atom: Chem.Atom) -> tuple[Any, ...]:
    return (
        atom.GetSymbol(),
        atom.GetFormalCharge(),
        atom.GetIsotope(),
        atom.GetIsAromatic(),
    )


def _atom_stereo(atom: Chem.Atom) -> str:
    if atom.HasProp("_CIPCode"):
        return atom.GetProp("_CIPCode")
    return str(atom.GetChiralTag())


def _bond_signature(bond: Chem.Bond) -> tuple[Any, ...]:
    return (str(bond.GetBondType()), bond.GetIsAromatic())


def _bond_stereo(bond: Chem.Bond) -> str:
    return str(bond.GetStereo())


def _difference(
    parent: Chem.Mol,
    child: Chem.Mol,
    parent_match: tuple[int, ...],
    child_match: tuple[int, ...],
) -> tuple[list[int], list[int], list[int], list[int]]:
    parent_to_child = dict(zip(parent_match, child_match))
    child_to_parent = dict(zip(child_match, parent_match))
    matched_child = set(child_match)
    matched_parent = set(parent_match)
    changed_atoms = {atom.GetIdx() for atom in child.GetAtoms()} - matched_child
    changed_bonds: set[int] = set()
    stereo_atoms: set[int] = set()
    stereo_bonds: set[int] = set()

    for parent_idx, child_idx in parent_to_child.items():
        parent_atom = parent.GetAtomWithIdx(parent_idx)
        child_atom = child.GetAtomWithIdx(child_idx)
        if _atom_signature(parent_atom) != _atom_signature(child_atom):
            changed_atoms.add(child_idx)
        if _atom_stereo(parent_atom) != _atom_stereo(child_atom):
            stereo_atoms.add(child_idx)

    common_child_bonds: set[int] = set()
    for parent_bond in parent.GetBonds():
        parent_begin = parent_bond.GetBeginAtomIdx()
        parent_end = parent_bond.GetEndAtomIdx()
        if parent_begin not in parent_to_child or parent_end not in parent_to_child:
            continue
        child_bond = child.GetBondBetweenAtoms(
            parent_to_child[parent_begin],
            parent_to_child[parent_end],
        )
        if child_bond is None:
            continue
        common_child_bonds.add(child_bond.GetIdx())
        if _bond_signature(parent_bond) != _bond_signature(child_bond):
            changed_bonds.add(child_bond.GetIdx())
            changed_atoms.update((child_bond.GetBeginAtomIdx(), child_bond.GetEndAtomIdx()))
        if _bond_stereo(parent_bond) != _bond_stereo(child_bond):
            stereo_bonds.add(child_bond.GetIdx())
            stereo_atoms.update((child_bond.GetBeginAtomIdx(), child_bond.GetEndAtomIdx()))

    for child_bond in child.GetBonds():
        if child_bond.GetIdx() not in common_child_bonds:
            changed_bonds.add(child_bond.GetIdx())
            changed_atoms.update((child_bond.GetBeginAtomIdx(), child_bond.GetEndAtomIdx()))

    # A deletion has no child-only atom. Highlight the attachment boundary.
    if not changed_atoms and len(parent_match) < parent.GetNumHeavyAtoms():
        for child_idx, parent_idx in child_to_parent.items():
            if any(nb.GetIdx() not in matched_parent for nb in parent.GetAtomWithIdx(parent_idx).GetNeighbors()):
                changed_atoms.add(child_idx)

    # Give atom-only substitutions and deletion boundaries a visible bond ribbon.
    if changed_atoms and not changed_bonds:
        for bond in child.GetBonds():
            if bond.GetBeginAtomIdx() in changed_atoms or bond.GetEndAtomIdx() in changed_atoms:
                changed_bonds.add(bond.GetIdx())

    return (
        sorted(changed_atoms),
        sorted(changed_bonds),
        sorted(stereo_atoms),
        sorted(stereo_bonds),
    )


def _draw_svg(
    mol: Chem.Mol,
    *,
    changed_atoms: Iterable[int] = (),
    changed_bonds: Iterable[int] = (),
    stereo_atoms: Iterable[int] = (),
    stereo_bonds: Iterable[int] = (),
    size: tuple[int, int] = SVG_SIZE,
) -> str:
    changed_atom_set = set(changed_atoms)
    changed_bond_set = set(changed_bonds)
    stereo_atom_set = set(stereo_atoms)
    stereo_bond_set = set(stereo_bonds)
    highlighted_atoms = sorted(changed_atom_set | stereo_atom_set)
    highlighted_bonds = sorted(changed_bond_set | stereo_bond_set)
    atom_colors = {
        idx: STEREO_COLOR if idx in stereo_atom_set else DIFF_COLOR
        for idx in highlighted_atoms
    }
    bond_colors = {
        idx: STEREO_BOND_COLOR if idx in stereo_bond_set else DIFF_BOND_COLOR
        for idx in highlighted_bonds
    }
    radii = {idx: 0.55 for idx in highlighted_atoms}

    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.fillHighlights = True
    options.highlightRadius = 0.55
    options.highlightBondWidthMultiplier = 10
    options.bondLineWidth = 2.1
    options.padding = 0.06
    options.minFontSize = 10 if size == TOPOLOGY_SVG_SIZE else 14
    options.maxFontSize = 18 if size == TOPOLOGY_SVG_SIZE else 24
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlighted_atoms,
        highlightBonds=highlighted_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
        highlightAtomRadii=radii,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.find("<svg") :].strip()


def _descriptors(mol: Chem.Mol) -> dict[str, Any]:
    return {
        "mw": round(Descriptors.MolWt(mol), 2),
        "clogp": round(Crippen.MolLogP(mol), 3),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "fraction_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
    }


def build_assets(payload: dict[str, Any]) -> dict[str, Any]:
    graph = payload["search_graph"]
    nodes = {str(node["id"]): node for node in graph.get("nodes", [])}
    root_id = str(graph["root_id"])
    root_node = nodes[root_id]
    root_mol = _parse(root_node["smiles"])
    rdDepictor.Compute2DCoords(root_mol)

    result: dict[str, Any] = {
        "schema_version": "m3os-molecule-diff-assets-v2",
        "legend": {
            "orange": "atoms and bonds changed from the direct parent",
            "purple": "stereochemistry changed from the direct parent",
        },
        "root": {
            "node_id": root_id,
            "smiles": root_node["smiles"],
            "svg": _draw_svg(root_mol),
            "topology_svg": _draw_svg(root_mol, size=TOPOLOGY_SVG_SIZE),
            "descriptors": _descriptors(root_mol),
        },
        "candidates": {},
    }

    for node_id, node in nodes.items():
        if node_id == root_id or node.get("is_root"):
            continue
        parent_ids = [str(item) for item in node.get("parent_ids", []) if str(item) in nodes]
        if not parent_ids:
            parent_ids = [root_id]
        comparisons = []
        for parent_id in parent_ids:
            parent_node = nodes[parent_id]
            parent = _parse(parent_node["smiles"])
            child = _parse(node["smiles"])
            parent_match, child_match, method = _mcs_matches(parent, child)
            _align(parent, child, parent_match, child_match)
            changed_atoms, changed_bonds, stereo_atoms, stereo_bonds = _difference(
                parent,
                child,
                parent_match,
                child_match,
            )
            comparisons.append(
                {
                    "parent_id": parent_id,
                    "parent_smiles": parent_node["smiles"],
                    "method": method,
                    "mcs_atom_count": len(child_match),
                    "changed_atom_count": len(changed_atoms),
                    "changed_bond_count": len(changed_bonds),
                    "stereo_changed_atom_count": len(stereo_atoms),
                    "stereo_changed_bond_count": len(stereo_bonds),
                    "parent_svg": _draw_svg(parent),
                    "svg": _draw_svg(
                        child,
                        changed_atoms=changed_atoms,
                        changed_bonds=changed_bonds,
                        stereo_atoms=stereo_atoms,
                        stereo_bonds=stereo_bonds,
                    ),
                }
            )
        child_for_props = _parse(node["smiles"])
        rdDepictor.Compute2DCoords(child_for_props)
        result["candidates"][node_id] = {
            "node_id": node_id,
            "smiles": node["smiles"],
            "primary_parent_id": comparisons[0]["parent_id"],
            "topology_svg": _draw_svg(child_for_props, size=TOPOLOGY_SVG_SIZE),
            "descriptors": _descriptors(child_for_props),
            "comparisons": comparisons,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    assets = build_assets(payload)
    args.output.write_text(
        json.dumps(assets, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output} with {len(assets['candidates'])} parent-aligned candidate depictions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
