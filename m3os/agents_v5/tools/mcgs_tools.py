"""Thin workflow-facing wrappers around the M3OS MCGS graph."""

from typing import Any, Dict, List

from m3os.agents_v5.tools.monte_carlo_graph.graph_for_molecular_optimization_v3_4 import MolecularGraph


def mcgs_add_root_node(
    mol_graph: MolecularGraph,
    root_smiles: str,
    intrinsic_score: float,
    properties: Dict[str, Any],
) -> MolecularGraph:
    """Add the root molecule and advance the graph iteration counter."""
    mol_graph.add_root_node(
        smiles=root_smiles,
        intrinsic_score=intrinsic_score,
        properties=properties,
    )
    mol_graph.update_iteration_count()
    return mol_graph


def mcgs_get_top_5_nodes(mol_graph: MolecularGraph):
    """Return the top five expandable graph nodes and the mutated graph."""
    return mol_graph.get_top_k_nodes_with_paths(k=5), mol_graph


def mcgs_add_optimized_molecules(
    mol_graph: MolecularGraph,
    parent_node_smiles: str,
    new_molecules_json: List[Dict[str, Any]],
    parent_node_select_reason: str,
) -> MolecularGraph:
    """Add evaluated child molecules under the selected parent node."""
    mol_graph.add_optimized_molecules(
        parent_node_smiles=parent_node_smiles,
        new_molecules_json=new_molecules_json,
        parent_node_select_reason=parent_node_select_reason,
    )
    return mol_graph


def mcgs_backpropagate(
    mol_graph: MolecularGraph,
    new_molecules_json: List[Dict[str, Any]],
    selected_node_smiles: str,
) -> MolecularGraph:
    """Backpropagate the best candidate score through the selected search path."""
    best_score = mol_graph.get_best_score(new_molecules_json=new_molecules_json)
    mol_graph.backpropagate_along_selected_path(
        selected_node_smiles=selected_node_smiles,
        reward=best_score,
    )
    mol_graph.update_all_uct_values()
    mol_graph.update_iteration_count()
    return mol_graph


def mcgs_visualize_graph(mol_graph: MolecularGraph) -> bool:
    """Render and save the current graph visualization."""
    mol_graph.visualize_graph(show_uct=True, show_scores=True)
    return True


def mcgs_get_node_num(mol_graph: MolecularGraph) -> int:
    """Return the total number of graph nodes."""
    return len(mol_graph.get_all_nodes())


def mcgs_is_next_round_required(mol_graph: MolecularGraph, node_num_needed: int) -> str:
    """Return whether graph expansion should continue."""
    return "False" if mcgs_get_node_num(mol_graph) >= node_num_needed else "True"


def mcgs_export_graph_to_csv(mol_graph: MolecularGraph) -> bool:
    """Export graph nodes to CSV, including the root molecule."""
    return mol_graph.save_nodes_to_csv(include_root=True)


__all__ = [
    "mcgs_add_root_node",
    "mcgs_get_top_5_nodes",
    "mcgs_add_optimized_molecules",
    "mcgs_backpropagate",
    "mcgs_visualize_graph",
    "mcgs_get_node_num",
    "mcgs_is_next_round_required",
    "mcgs_export_graph_to_csv",
]
