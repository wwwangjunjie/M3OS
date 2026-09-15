"""Functional-group fragmentation helpers."""

from __future__ import annotations

import ast
from typing import Dict, List

from accfg import AccFG


def smiles2fragments(smiles: str) -> str:
    """
    Fragment a molecule from its SMILES and return the AccFG result as a string.

    Args:
        smiles: The SMILES representation of the molecule.

    Returns:
        String representation of a dictionary of functional groups.
    """
    afg = AccFG(print_load_info=False)
    fgs, _fg_graph = afg.run(smiles, show_atoms=True, show_graph=True)
    return str(fgs)


def smiles2fragments_str(smiles_list: List[str]) -> Dict[str, str]:
    """
    Fragment multiple molecules into functional-group names.

    Args:
        smiles_list: Input molecular SMILES strings.

    Returns:
        Mapping from each input SMILES to a stringified list of functional-group names,
        or to the error message produced while processing that molecule.
    """
    result: Dict[str, str] = {}

    for smiles in smiles_list:
        try:
            fgs_tobe_opt = smiles2fragments(smiles)
            fgs_tobe_opt = ast.literal_eval(fgs_tobe_opt)
            fgs_tobe_opt_list = list(fgs_tobe_opt.keys())
            result[smiles] = str(fgs_tobe_opt_list)
        except Exception as exc:
            result[smiles] = str(exc)

    return result
