# https://github.com/markotot/MonteCarloGraphSearch/blob/main/Agents/MCGS/Graph.py

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from typing import Dict, List, Optional, Any, Set, Tuple
import hashlib
from collections import defaultdict
import pickle
import csv
import json
import os
from datetime import datetime
from pathlib import Path

from configs.settings import DEFAULT_ENV_FILE, get_setting, load_settings

plt.style.use('default')

M3OS_ROOT = Path(__file__).resolve().parents[4]
MCG_RUNTIME_OUTPUT_DIR = M3OS_ROOT / "tmp"
_SETTINGS = load_settings(
    os.getenv("M3OS_ENV_FILE") or DEFAULT_ENV_FILE,
    use_default_config=True,
)
CSV_SAVE_PATH_SUFFIX = str(get_setting(_SETTINGS, "CSV_SAVE_PATH_SUFFIX", "mcgs.csv_save_path_suffix", "110"))


def _resolve_generated_file_path(filepath: str, base_dir: str) -> str:
    path = Path(filepath).expanduser()
    if path.is_absolute():
        return str(path)
    return str((Path(base_dir) / path).resolve())

# TODO: Keep a node unreachable until it has a reachable parent connected to the root; update top-k selection accordingly.
# TODO: Define backpropagation behavior for paths containing unreachable nodes.
# TODO: Recompute graph-wide UCT values when node reachability changes.
# TODO: Revisit UCT calculation when a backpropagated node has an unreachable parent.
# TODO: Consider retaining backpropagation history for later UCT recalculation.


class MoleculeNode:
    """Molecule node containing chemical information and search state."""
    def __init__(
        self, 
        smiles: str,
        intrinsic_score: float = 0.0, 
        properties: Optional[Dict] = None,
        iteration: int = 0,  # Iteration 0 denotes the root node.
        agent_type: str = None,
    ):
        """
        Args:
            smiles: Molecular SMILES string.
            intrinsic_score: Critic's initial score; not changed by backpropagation.
            properties: Molecular property mapping.
            iteration: Iteration in which the node was added.
        """
        self.smiles = smiles
        self.properties = properties or {}
        
        self.id = self._generate_id(smiles)
        
        # Graph search state.
        self.parents: List[MoleculeNode] = []  # Multiple parent nodes.
        self.actions_from_parents: Dict[str, str] = {}  # Transformation for each parent ID.

        # Node-value components.
        self.intrinsic_score = intrinsic_score  # Critic's immutable initial score.
        self.total_reward = 0.0 + self.intrinsic_score                # Accumulated backpropagated reward.
        self.visit_count = 0                   # Number of visits, n.
        self.uct_value = 0.0                   # Dynamically computed UCT score.

        self.unreachable = False
        # self.depth = 0  # Minimum depth from the root.
        
        # Molecule-specific information.
        self.agent_type = agent_type  # Agent that generated this molecule.

        self.rationale_from_parents_generator: Dict[str, str] = {}  # Generator rationale by parent ID.
        self.rationale_from_parents_critic: Dict[str, str] = {}  # Critic rationale by parent ID.

        self.confidence_score_from_parents_generator: Dict[str, float] = {}  # Generator confidence by parent ID.

        # Record the iteration when this node was added.
        self.iteration = iteration

        # Store the reason for selecting this node for optimization.
        self.selected_reason: List[Dict[int, str]] = []  # Format: [{iteration: reason}].
        

    def _generate_id(self, smiles: str) -> str:
        """Generate a unique ID from SMILES."""
        return f"MOL_{hashlib.md5(smiles.encode()).hexdigest()[:16]}"
    
    def intrinsic_score(self) -> float:
        """Return the score assigned by the Critic agent."""
        return self.intrinsic_score

    def average_reward(self) -> float:
        """Return the exploitation term in the UCB formula."""
        exploitation = self.total_reward / self.visit_count if self.visit_count !=0 else self.total_reward
        return exploitation
    
    def update_node(self, reward: float):
        """Update node statistics during backpropagation."""
        self.visit_count += 1
        self.total_reward += reward

    # def add_parent(self, parent: 'MoleculeNode', action: str = None):
    #     """Add a parent node."""
    #     if parent.id not in [p.id for p in self.parents]:
    #         self.parents.append(parent)
    #         self.actions_from_parents[parent.id] = action
    #         # Update depth to the minimum parent depth plus one.
    #         # self.depth = min(self.depth, parent.depth + 1) if self.parents else parent.depth + 1

    def to_dict(self) -> Dict:
        """Convert the node to a dictionary."""
        return {
            'id': self.id,
            'smiles': self.smiles,
            'properties': self.properties,
            'intrinsic_score': self.intrinsic_score,
            'total_reward': self.total_reward,
            'visit_count': self.visit_count,
            'uct_value': self.uct_value,
            # 'depth': self.depth,
            # 'agent_type': self.agent_type,
            'rationale': self.rationale,
            'parent_count': len(self.parents),
            'parent': [parent.smiles for parent in self.parents],
            'actions_from_parents': self.actions_from_parents,
            'rationale_from_parents_critic': self.rationale_from_parents_critic,
            'rationale_from_parents_generator': self.rationale_from_parents_generator,
            'confidence_score_from_parents_generator': self.confidence_score_from_parents_generator,
            'unreachable': self.unreachable,
            'iteration': self.iteration,
        }


class MoleculeEdge:
    """Edge representing a molecular transformation."""
    def __init__(
        self, 
        node_from: MoleculeNode, 
        node_to: MoleculeNode,  
        action: str, 
    ):
        self.node_from = node_from
        self.node_to = node_to
        self.action = action

        # TODO: Add generator rationale to the edge.


class MolecularGraph:
    """
    Molecular optimization graph manager using Monte Carlo graph search.
    """
    
    def __init__(
        self, 
        seed: int, 
        exploration_weight: float = 0.01  # UCT exploration weight.
    ):
        """
        Args:
            seed: Random seed.
            exploration_weight: UCT exploration-weight parameter.
        """
        self.graph = nx.DiGraph()
        self.root_node = None
        self.exploration_weight = exploration_weight
        
        # Molecule-specific lookup caches.
        self.smiles_to_node = {}  # Map SMILES to nodes.
        self.node_id_to_node = {}  # Map IDs to nodes.
        
        # Search state.
        self.iteration_count = 0
        self.total_simulations = 0
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Candidate-path storage.
        self.candidate_paths = {}  # Shortest path by candidate SMILES.
        self.selected_path = []  # Currently selected path: [(node, action)].

        # Set the random seed.
        np.random.seed(seed)

    @staticmethod
    def _get_csv_fieldnames() -> List[str]:
        return [
            'node_id',
            'smiles',
            'intrinsic_score',
            'total_reward',
            'visit_count',
            'uct_value',
            'unreachable',
            'parent_count',
            'parent_smiles_list',
            'actions_from_parents',
            'critic_rationales_from_parents',
            'generator_rationales_from_parents',
            'generator_confidence_from_parents',
            'properties',
            'iteration_count',
            'selected_reason',
            'agent_type'
        ]

    def _resolve_output_filename(self, filename: str, tmp: bool = False) -> str:
        if filename == 'graph_nodes.csv' and self.root_node:
            filename = f"{self.root_node.smiles}_{self.run_timestamp}_{CSV_SAVE_PATH_SUFFIX}.csv"
        if tmp:
            base, ext = os.path.splitext(filename)
            ext = ext if ext else ".csv"
            filename = f"{base}.tmp{ext}"
        return filename

    def _node_to_csv_row(self, node: MoleculeNode) -> Dict[str, Any]:
        parents = self.get_parents(node)

        parent_smiles_list = [p.smiles for p in parents] if parents else []

        actions_list = [
            {f"{self.get_node_by_id(pid).smiles if self.get_node_by_id(pid) else pid}": act}
            for pid, act in node.actions_from_parents.items()
        ] if node.actions_from_parents else []

        critic_rationales_list = [
            {f"{self.get_node_by_id(pid).smiles if self.get_node_by_id(pid) else pid}": rat}
            for pid, rat in node.rationale_from_parents_critic.items()
        ] if node.rationale_from_parents_critic else []

        generator_rationales_list = [
            {f"{self.get_node_by_id(pid).smiles if self.get_node_by_id(pid) else pid}": rat}
            for pid, rat in node.rationale_from_parents_generator.items()
        ] if node.rationale_from_parents_generator else []

        generator_confidence_list = [
            {f"{self.get_node_by_id(pid).smiles if self.get_node_by_id(pid) else pid}": conf}
            for pid, conf in node.confidence_score_from_parents_generator.items()
        ] if node.confidence_score_from_parents_generator else []

        return {
            'node_id': node.id,
            'smiles': node.smiles,
            'intrinsic_score': f"{node.intrinsic_score:.4f}",
            'total_reward': f"{node.total_reward:.4f}",
            'visit_count': node.visit_count,
            'uct_value': f"{node.uct_value:.4f}",
            'unreachable': node.unreachable,
            'parent_count': len(parents),
            'parent_smiles_list': json.dumps(parent_smiles_list),
            'actions_from_parents': json.dumps(actions_list),
            'critic_rationales_from_parents': json.dumps(critic_rationales_list),
            'generator_rationales_from_parents': json.dumps(generator_rationales_list),
            'generator_confidence_from_parents': json.dumps(generator_confidence_list),
            'properties': json.dumps(node.properties),
            'iteration_count': node.iteration,
            'selected_reason': json.dumps(node.selected_reason),
            'agent_type': node.agent_type,
        }

    def save_nodes_to_tmp_csv(self, filename: str = 'graph_nodes.csv', include_root: bool = False) -> bool:
        """Overwrite the temporary CSV with a graph snapshot for recovery."""
        return self.save_nodes_to_csv(filename=filename, include_root=include_root, tmp=True)
    
    # ========== Graph structure management ==========
    
    def add_node(self, node: MoleculeNode):
        """Add a node to the graph."""
        self.graph.add_node(node.id, info=node)
        self.smiles_to_node[node.smiles] = node
        self.node_id_to_node[node.id] = node
    
    def add_edge(self, edge: MoleculeEdge):
        """Add an edge to the graph."""
        # Add the edge to the graph.
        self.graph.add_edge(edge.node_from.id, edge.node_to.id, info=edge)
        
        # Update the child node's parent information.
        # edge.node_to.add_parent(edge.node_from, edge.action)
    
    def set_root_node(self, root_node_smiles: str):
        """Set the root node."""
        root_node = self.get_node_by_smiles(root_node_smiles)
        self.root_node = root_node
        # root_node.depth = 0
    
    def add_root_node(self, smiles: str, intrinsic_score: float = 0.0, properties: Optional[Dict] = None):
        """Add the root node."""
        root_node = MoleculeNode(
            smiles=smiles, 
            intrinsic_score=intrinsic_score, 
            properties=properties, 
            iteration=0,
            agent_type="human_insert",
        )  # The root belongs to iteration 0.
        self.add_node(root_node)
        self.set_root_node(root_node.smiles)
    
    def get_node_by_id(self, node_id: str) -> Optional[MoleculeNode]:
        """Get a node by ID."""
        return self.node_id_to_node.get(node_id)
    
    def get_node_by_smiles(self, smiles: str) -> Optional[MoleculeNode]:
        """Get a node by SMILES."""
        return self.smiles_to_node.get(smiles)
    
    def has_node(self, node_id: str) -> bool:
        """Check whether a node exists."""
        return self.graph.has_node(node_id)
    
    def has_edge(self, node_from: MoleculeNode, node_to: MoleculeNode) -> bool:
        """Check whether two nodes are connected by an edge."""
        return self.graph.has_edge(node_from.id, node_to.id)
    
    def get_children(self, node: MoleculeNode) -> List[MoleculeNode]:
        """Get all child nodes of a node."""
        children = []
        for child_id in self.graph.successors(node.id):
            child_node = self.get_node_by_id(child_id)
            if child_node:
                children.append(child_node)
        return children
    
    def get_parents(self, node: MoleculeNode) -> List[MoleculeNode]:
        """Get all parent nodes of a node."""
        parents = []
        for parent_id in self.graph.predecessors(node.id):
            parent_node = self.get_node_by_id(parent_id)
            if parent_node:
                parents.append(parent_node)
        return parents
    
    def get_all_nodes(self) -> List[MoleculeNode]:
        """Get all graph nodes."""
        return list(self.node_id_to_node.values())
    
    # ========== UCT calculation ==========
    
    def compute_uct_value(self, node: MoleculeNode) -> float:
        """
        Calculate a node's graph-level UCT value.
        
        Parent visit count is the sum of all of this node's parents' visits.
        
        Args:
            node: Node whose UCT value is calculated.
            
        Returns:
            UCT value.
        """
        # TODO: Review exploration when one selected node produces five children per iteration.
        # if node.visit_count == 0:
        #     return float('inf')  # Prioritize unvisited nodes.
        
        # Sum the visits of all parent nodes.
        parents = self.get_parents(node)
        if parents:
            parent_total_visits = sum(p.visit_count for p in parents)
        else:
            # For a parentless node such as the root, use its own visit count.
            parent_total_visits = max(node.visit_count, 1)
        
        # UCT formula: Q + c * sqrt(ln(N) / n).
        # Use the node's aggregate value as Q.
        exploitation = node.average_reward()
        if node.visit_count !=0:
            exploration = self.exploration_weight * np.sqrt(np.log(parent_total_visits) / node.visit_count)
        else:
            exploration = self.exploration_weight * np.sqrt(np.log(parent_total_visits) / 1)
        
        return exploitation + exploration
    

    def update_all_uct_values(self):
        """
        Update the UCT value of every node in the graph.
        Call this before selecting candidate nodes.
        """
        for node in self.get_all_nodes():
            node.uct_value = self.compute_uct_value(node)

    
    # def get_top_k_nodes_with_paths(self, k: int = 5, only_reachable: bool = True) -> Tuple[List[str], Dict[str, Tuple[List[str], List[str]]]]:
    def get_top_k_nodes_with_paths(self, k: int = 5, only_reachable: bool = True) -> List[str]:
        """
        Select the k highest-UCT nodes and save their shortest root paths.
        
        Args:
            k: Number of nodes to return.
            only_reachable: Whether to consider only nodes reachable from the root.
            
        Returns:
            (candidate SMILES list, path map from SMILES to path nodes and actions).
        """
        # Update every node's UCT value.
        self.update_all_uct_values()
        
        all_nodes = self.get_all_nodes()
        
        # Select reachable nodes based on path reachability.
        # if only_reachable and self.root_node:
        #     # Filter nodes reachable from the root.
        #     reachable_nodes = []
        #     for node in all_nodes:
        #         if node == self.root_node:
        #             reachable_nodes.append(node)
        #         elif nx.has_path(self.graph, self.root_node.id, node.id):
        #             reachable_nodes.append(node)
        #     candidates = reachable_nodes
        # else:
        #     candidates = all_nodes

        # Filter nodes using the unreachable flag.
        if only_reachable and self.root_node:
            # Keep nodes whose unreachable flag is false.
            candidates = [node for node in all_nodes if not node.unreachable]
        else:
            candidates = all_nodes.copy()
        
        # Exclude the root node.
        if len(candidates) > 1:
            if self.root_node and self.root_node in candidates:
                candidates.remove(self.root_node)
        
        if not candidates:
            return [], {}
        
        # Sort by UCT value.
        candidates.sort(key=lambda x: x.uct_value, reverse=True)
        top_k_nodes = candidates[:k]
        top_k_nodes_smiles = [node.smiles for node in candidates[:k]]
        
        # Save each candidate's shortest path.
        self.candidate_paths = {}
        for node in top_k_nodes:
            # Find all paths from the root to this node.
            try:
                # Enumerate simple paths.
                all_paths = list(nx.all_simple_paths(self.graph, self.root_node.id, node.id))
                
                if all_paths:
                    # Select the path with the fewest nodes.
                    shortest_path = min(all_paths, key=len)
                    
                    # Convert the path to node objects and actions.
                    path_nodes = [self.get_node_by_id(node_id) for node_id in shortest_path]
                    path_nodes_smiles = [node.smiles for node in path_nodes]
                    
                    # Collect actions along the path.
                    path_actions = ['']
                    for i in range(len(shortest_path) - 1):
                        edge_data = self.graph.get_edge_data(shortest_path[i], shortest_path[i + 1])
                        if edge_data and 'info' in edge_data:
                            path_actions.append(edge_data['info'].action)
                        else:
                            path_actions.append("Unknown")
                    
                    self.candidate_paths[node.smiles] = (path_nodes_smiles, path_actions)
                    
                    print(f"Node {node.smiles} (UCT: {node.uct_value:.3f}): "
                          f"Path length = {len(path_nodes)-1}")
                else:
                    print(f"Warning: No path found for node {node.smiles}")
                    
            except nx.NetworkXNoPath:
                print(f"Warning: No path from root to node {node.smiles}")
            except nx.NodeNotFound:
                print(f"Warning: Node {node.smiles} not found in graph")
        
        # return top_k_nodes_smiles, self.candidate_paths
        return top_k_nodes_smiles


    def select_expansion_node_with_llm(self, k: int = 5) -> Tuple[MoleculeNode, List[MoleculeNode], List[str]]:
        """
        Select the k highest-UCT molecules, save their paths, and simulate an LLM choice.
        
        Args:
            k: Number of candidate nodes.
            
        Returns:
            (selected node, path node list, path action list).
        """
        # Obtain candidate nodes and paths.
        candidate_nodes, candidate_paths = self.get_top_k_nodes_with_paths(k, only_reachable=True)
        
        if not candidate_nodes:
            if self.root_node:
                # Return the root if it is the only available node.
                return self.root_node, [self.root_node], ["ROOT"]
            else:
                raise ValueError("No nodes available for selection")

        # TODO: Return JSON containing candidate-node metadata.
        
        print(f"\nTop {k} candidate nodes (by UCT):")
        for i, node in enumerate(candidate_nodes, 1):
            path_info = candidate_paths.get(node.id, ([], []))
            path_nodes, path_actions = path_info
            print(f"{i}. Node {node.id}")
            print(f"   SMILES: {node.smiles[:50]}...")
            print(f"   UCT: {node.uct_value:.3f}, Value: {node.value():.3f}, Visits: {node.visit_count}")
            print(f"   Path length: {len(path_nodes)-1}")
            if node.parents:
                print(f"   Parent count: {len(node.parents)}, Parent visits sum: {sum(p.visit_count for p in node.parents)}")
            print()
        
        # Simulate the LLM selection process.
        # A production implementation should pass candidate metadata to an LLM API.
        # For now, choose the node with the highest UCT value.
        selected_node = candidate_nodes[0]
        
        # Obtain the selected path.
        if selected_node.id in self.candidate_paths:
            selected_path_nodes, selected_path_actions = self.candidate_paths[selected_node.id]
            self.selected_path = list(zip(selected_path_nodes, selected_path_actions))
        else:
            # Find a shortest path if none was saved.
            selected_path_nodes, selected_path_actions = self.get_shortest_path_from_root(selected_node)
            self.selected_path = list(zip(selected_path_nodes, selected_path_actions))
        
        print(f"LLM selected node: {selected_node.id}")
        print(f"Selected path length: {len(selected_path_nodes)-1}")
        
        return selected_node, selected_path_nodes, selected_path_actions

    # TODO: Accept an LLM-selected SMILES, choose its search path, and update state.
    

    def get_shortest_path_from_root(self, node_smiles: str) -> Tuple[List[str], List[str]]:
        """
        Find the shortest path from the root to the target node.
        
        Returns:
            nodes_smiles: SMILES of nodes along the path.
            actions: List of transformations along the path.
        """
        if not self.root_node:
            return [], []
        
        node = self.get_node_by_smiles(node_smiles) if isinstance(node_smiles, str) else None
        
        try:
            # Use NetworkX's shortest-path algorithm.
            node_ids = nx.shortest_path(self.graph, self.root_node.id, node.id)
            
            nodes = [self.get_node_by_id(node_id) for node_id in node_ids]
            nodes_smiles = [node_in_path.smiles for node_in_path in nodes]

            actions = []
            
            # Collect actions along the path.
            for i in range(len(node_ids) - 1):
                edge_data = self.graph.get_edge_data(node_ids[i], node_ids[i + 1])
                if edge_data and 'info' in edge_data:
                    actions.append(edge_data['info'].action)
                else:
                    actions.append("Unknown")
            
            return nodes_smiles, actions
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], []
        
    
    def get_best_score(self, new_molecules_json: List[Dict[str, Any]],) -> float:
        """
        Find the highest intrinsic score in a list of optimized molecules.
        
        Args:
            new_molecules_json format: [{"smiles": str, "action": str, "critic_rationale": str, "generator_rationale": str, "generator_confidence_score": float, "score": float, "agent_type": str, "properties": dict}, ...]
            
        Returns:
            Highest intrinsic score.
        """
        if not new_molecules_json:
            return 0.0
        
        # 1. Convert dictionaries to the tuples required by the helper.
        optimized_molecules = [
            (m['smiles'], m['action'], m['critic_rationale'], m['score'], m['agent_type'], m['properties'])
            for m in new_molecules_json
        ]
        
        intrinsic_scores = [score for _, _, _, score, _, _ in optimized_molecules]
        max_intrinsic_score = max(intrinsic_scores)
        
        return max_intrinsic_score

    def backpropagate_along_selected_path(self, selected_node_smiles:str, reward: float):
        """
        Backpropagate reward along the selected path.
        
        Args:
            reward: Value to propagate (highest intrinsic score among five molecules).
        """
        # selected_path_ = self.candidate_paths[selected_node_smiles]
        selected_node = self.get_node_by_smiles(selected_node_smiles)
        
        print("selected_node!!!:", selected_node)

        # Obtain the selected path.
        if selected_node_smiles in self.candidate_paths:
            selected_path_nodes, selected_path_actions = self.candidate_paths[selected_node_smiles]
            self.selected_path = list(zip(selected_path_nodes, selected_path_actions))
        else:
            # Find a shortest path if none was saved.
            selected_path_nodes, selected_path_actions = self.get_shortest_path_from_root(selected_node_smiles)
            self.selected_path = list(zip(selected_path_nodes, selected_path_actions))
        

        if not self.selected_path:
            print("Warning: No selected path. Cannot backpropagate.")
            return False
        
        print(f"\nBackpropagating reward {reward:.3f} along selected path...")
        
        # Propagate from the leaf node toward the root.
        path_nodes = [self.get_node_by_smiles(node_smiles_in_path) for node_smiles_in_path, _ in self.selected_path]
        
        # Backpropagate reward.
        for i in range(len(path_nodes)-1, -1, -1):
            node = path_nodes[i]
            
            # Update node statistics.
            node.visit_count += 1
            node.total_reward += reward
            # node.cumulative_value = node.intrinsic_score + node.total_reward
            
            # TODO: Reconsider calculation before parent visit counts are updated.
            # Recompute UCT using the sum of all parent visit counts.
            # node.uct_value = self.compute_uct_value(node)
            
            print(f"  Updated node {node.smiles}: visits={node.visit_count}, "
                  f"total_reward={node.total_reward:.3f}, value={node.average_reward():.3f}")
        
        print("Backpropagation complete.\n")

        self.update_all_uct_values()
        print("Updated UCT values for all nodes after backpropagation.")

        return True


    
 
    def add_optimized_molecules(
            self, 
            parent_node_smiles: str,
            # optimized_molecules: List[Tuple[str, str, str, float, Dict]],
            new_molecules_json: List[Dict[str, Any]],
            parent_node_select_reason: str = None,
        ) -> bool:
        """
        Add optimized molecules to the graph.
        
        Args:
            parent_node: Parent node that was optimized.
            new_molecules_json format: [{"smiles": str, "action": str, "critic_rationale": str, "generator_rationale": str, "generator_confidence_score": float, "score": float, "agent_type": str, "properties": dict}, ...]

        Returns:
            List of added nodes.
        """

        parent_node = self.smiles_to_node.get(parent_node_smiles)
        if parent_node is None:
            # Fall back to the root if the supplied parent SMILES is absent.
            fallback_node = getattr(self, "root_node", None)
            if fallback_node is None:
                print(
                    f"Warning: parent node '{parent_node_smiles}' not found and root_node is unavailable. "
                    "Skip add_optimized_molecules for this round."
                )
                return False
            print(
                f"Warning: parent node '{parent_node_smiles}' not found. "
                f"Fallback to root node '{fallback_node.smiles}'."
            )
            parent_node = fallback_node
        parent_node.selected_reason.append({self.iteration_count: parent_node_select_reason})

        added_nodes = []

        # 1. Convert dictionaries to the tuples required by the helper.
        optimized_molecules = [
            (m['smiles'], m['action'], m['critic_rationale'], m['generator_rationale'], m['generator_confidence_score'], m['score'], m['agent_type'], m['properties'])
            for m in new_molecules_json
        ]

        for smiles, action, critic_rationale, generator_rationale, generator_confidence_score, score, agent_type, properties in optimized_molecules:
            
            # Skip the root node.
            if smiles == self.root_node.smiles:
                continue

            # Check whether this molecule already exists.
            existing_node = self.get_node_by_smiles(smiles)
            
            if existing_node:
                # Add an edge for an existing molecule if needed.
                if not self.has_edge(parent_node, existing_node):
                    edge = MoleculeEdge(parent_node, existing_node, action)
                    self.add_edge(edge)
                                        
                    # Update the score if the new connection provides a different value.
                    if score != existing_node.intrinsic_score:
                        existing_node.total_reward += (score - existing_node.intrinsic_score)  # Reflect the revised score in total reward.
                        existing_node.intrinsic_score = score

                    existing_node.parents.append(parent_node)
                    existing_node.actions_from_parents[parent_node.smiles] = action
                    existing_node.rationale_from_parents_critic[parent_node.smiles] = critic_rationale
                    existing_node.rationale_from_parents_generator[parent_node.smiles] = generator_rationale
                    existing_node.confidence_score_from_parents_generator[parent_node.smiles] = generator_confidence_score
                    existing_node.agent_type = agent_type
                
                node = existing_node
            else:
                # Create a new node.
                node = MoleculeNode(
                    smiles=smiles, 
                    properties=properties,
                    intrinsic_score=score, 
                    iteration=self.iteration_count,  # Use the current iteration.
                    agent_type=agent_type
                )
                node.parents.append(parent_node)
                node.actions_from_parents[parent_node.smiles] = action
                node.rationale_from_parents_critic[parent_node.smiles] = critic_rationale
                node.rationale_from_parents_generator[parent_node.smiles] = generator_rationale
                node.confidence_score_from_parents_generator[parent_node.smiles] = generator_confidence_score
 
                # node.depth = parent_node.depth + 1
                
                # Add the node to the graph.
                self.add_node(node)
                
                # Add the edge.
                edge = MoleculeEdge(parent_node, node, action)
                self.add_edge(edge)
                
                # Record the search path.
                # self.current_search_path.append((node, action))
            
            added_nodes.append(node)

        # Optional debug/backup snapshot. Production M3OS exports CSV on user download.
        if added_nodes and os.environ.get("M3OS_MCGS_TMP_CSV", "").lower() in {"1", "true", "yes", "on"}:
            self.save_nodes_to_tmp_csv(include_root=False)

        return True


    def update_iteration_count(self):
        """Advance the iteration counter."""
        self.iteration_count = self.iteration_count + 1
    


    # def run_mcgs_iteration_with_agents(
    #     self,
    #     agent2_optimize_func,
    #     agent3_score_func,
    #     selection_k: int = 5,
    #     num_optimized: int = 5
    # ) -> Dict:
    #     """
    #     Run one complete multi-agent MCGS iteration.
        
    #     Args:
    #         agent2_optimize_func: Agent 2 optimizer; accepts SMILES and returns [(SMILES, rationale)].
    #         agent3_score_func: Agent 3 scorer; accepts SMILES and returns a score.
    #         selection_k: Number of candidate nodes to select.
    #         num_optimized: Number of molecules optimized per node.
            
    #     Returns:
    #         Iteration-result dictionary.
    #     """
    #     self.iteration_count += 1
    #     print(f"\n{'='*60}")
    #     print(f"MCGS Iteration {self.iteration_count}")
    #     print(f"{'='*60}")
        
    #     # Step 1: Agent 1 selects a node and path.
    #     selected_node, path_nodes, path_actions = self.select_expansion_node_with_llm(k=selection_k)
        
    #     # Step 2: Agent 2 optimizes the molecule.
    #     print(f"\nAgent 2: Optimizing molecule {selected_node.smiles[:30]}...")
    #     optimized_results = agent2_optimize_func(selected_node.smiles, num_optimized)
        
    #     if not optimized_results:
    #         print("Agent 2 failed to generate optimized molecules")
    #         return {'status': 'agent2_failed'}
        
    #     # Step 3: Agent 3 scores the molecule.
    #     print(f"\nAgent 3: Evaluating {len(optimized_results)} molecules...")
    #     scored_molecules = []
    #     intrinsic_scores = []
        
    #     for smiles, rationale in optimized_results:
    #         score = agent3_score_func(smiles)
    #         properties = {"score": score}
    #         scored_molecules.append((smiles, rationale, score, properties))
    #         intrinsic_scores.append(score)
            
    #         print(f"  - {smiles[:40]}...: Score={score:.3f}")
        
    #     # Step 4: Agent 1 adds optimized molecules to the graph.
    #     print(f"\nAgent 1: Adding optimized molecules to graph...")
    #     added_nodes = self.add_optimized_molecules(
    #         selected_node, 
    #         scored_molecules,
    #         # agent_type="Agent2"
    #     )
        
    #     # Step 5: Agent 1 backpropagates reward.
    #     print(f"\nAgent 1: Backpropagating rewards...")
    #     if added_nodes and intrinsic_scores:
    #         # Use the highest intrinsic score among five molecules as reward.
    #         max_intrinsic_score = max(intrinsic_scores)
    #         print(f"Using max intrinsic score as reward: {max_intrinsic_score:.3f}")
            
    #         # Backpropagate along the selected path.
    #         self.backpropagate_along_selected_path(max_intrinsic_score)
        
    #     # Update all UCT values for the next iteration.
    #     self.update_all_uct_values()
        
    #     # Get the best molecule.
    #     best_nodes = self.get_best_nodes_by_criteria("value", k=3)
        
    #     # Build the result.
    #     result = {
    #         'status': 'success',
    #         'iteration': self.iteration_count,
    #         'selected_node': selected_node.to_dict(),
    #         'selected_path_length': len(path_nodes) - 1,
    #         'optimized_count': len(added_nodes),
    #         'max_intrinsic_score': max(intrinsic_scores) if intrinsic_scores else 0,
    #         'best_nodes': [node.to_dict() for node in best_nodes],
    #         'average_score': np.mean(intrinsic_scores) if intrinsic_scores else 0
    #     }
        
    #     # Clear the selected path for the next iteration.
    #     self.selected_path = []
        
    #     return result


    def set_node_unreachable_with_descendants(self, node_smiles_or_id: str):
        """
        Mark a node and its descendants unreachable unless they have another reachable parent.
        
        Args:
            node_smiles_or_id: Node SMILES or node ID.
            
        Returns:
            List[str]: IDs of nodes marked unreachable.
        """
        # Get the target node.
        target_node = None
        if node_smiles_or_id in self.smiles_to_node:
            target_node = self.smiles_to_node[node_smiles_or_id]
        elif node_smiles_or_id in self.node_id_to_node:
            target_node = self.node_id_to_node[node_smiles_or_id]
        else:
            print(f"Error: Node {node_smiles_or_id} not found in graph")
            return []
        
        if target_node.unreachable:
            print(f"Node {target_node.smiles} is already unreachable")
            return []
        
        print(f"\n{'='*60}")
        print(f"Setting node {target_node.smiles[:30]}... and its descendants as unreachable")
        print(f"{'='*60}")
        
        # Traverse all descendants with breadth-first search.
        from collections import deque
        
        nodes_to_process = deque([target_node])
        processed_nodes = set()
        unreachable_nodes = []
        
        # First pass: mark every potentially affected node.
        all_descendants = set()
        queue = deque([target_node])
        
        while queue:
            current = queue.popleft()
            if current.id in all_descendants:
                continue
            all_descendants.add(current.id)
            
            # Get all child nodes.
            children = self.get_children(current)
            for child in children:
                if child.id not in all_descendants:
                    queue.append(child)
        
        print(f"Found {len(all_descendants)} potential nodes to evaluate (including target)")
        
        # Second pass: check each descendant for another reachable parent.
        # First collect all reachable nodes, starting from the root.
        reachable_nodes = set()
        if self.root_node:
            for node in self.get_all_nodes():
                try:
                    if node == self.root_node or nx.has_path(self.graph, self.root_node.id, node.id):
                        reachable_nodes.add(node.id)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
        
        # For each descendant of the target node.
        for descendant_id in all_descendants:
            descendant = self.get_node_by_id(descendant_id)
            if not descendant:
                continue
            
            # Get all parent nodes.
            parents = self.get_parents(descendant)
            
            # Check for a reachable parent outside the target subgraph.
            has_other_reachable_parent = False
            alternative_parents = []
            
            for parent in parents:
                # The parent must not be a target descendant and must be reachable.
                if parent.id not in all_descendants and parent.id in reachable_nodes:
                    has_other_reachable_parent = True
                    alternative_parents.append(parent)
            
            if has_other_reachable_parent:
                print(f"  Node {descendant.smiles[:30]}... remains reachable via alternative parents: "
                    f"{[p.smiles[:20] for p in alternative_parents]}")
            else:
                # No alternative reachable parent: mark this node unreachable.
                descendant.unreachable = True
                unreachable_nodes.append(descendant.smiles)
                print(f"  Setting {descendant.smiles[:30]}... as unreachable")
        
        target_node.unreachable = True
        unreachable_nodes.insert(0, target_node.smiles) 

        print(f"\nSet {len(unreachable_nodes)} nodes as unreachable")
        
        # Update all UCT values.
        self.update_all_uct_values()
        
        return unreachable_nodes


    def set_node_reachable(self, node_smiles_or_id: str):
        """
        Mark a specified node reachable again.
        
        Args:
            node_smiles_or_id: Node SMILES or node ID.
            
        Returns:
            bool: Whether the node was successfully made reachable.
        """
        # Get the target node.
        target_node = None
        if node_smiles_or_id in self.smiles_to_node:
            target_node = self.smiles_to_node[node_smiles_or_id]
        elif node_smiles_or_id in self.node_id_to_node:
            target_node = self.node_id_to_node[node_smiles_or_id]
        else:
            print(f"Error: Node {node_smiles_or_id} not found in graph")
            return False
        
        if not target_node.unreachable:
            print(f"Node {target_node.smiles} is already reachable")
            return True
        else:
            target_node.unreachable = False  # Temporarily mark reachable to inspect parents.
            return True
        
        # Check whether this node has a reachable parent.
        parents = self.get_parents(target_node)
        has_reachable_parent = False
        
        for parent in parents:
            if not parent.unreachable:
                has_reachable_parent = True
                break
        
        if has_reachable_parent:
            target_node.unreachable = False
            print(f"Node {target_node.smiles[:30]}... set as reachable")
            self.update_all_uct_values()
            return True
        else:
            print(f"Cannot set node as reachable: no reachable parent found")
            return False


        
    
    def to_dict(self) -> Dict:
        """Convert the graph to a dictionary."""
        all_nodes = self.get_all_nodes()
        
        nodes_dict = {}
        for node in all_nodes:
            nodes_dict[node.id] = node.to_dict()
        
        edges_dict = []
        for u, v, data in self.graph.edges(data=True):
            if 'info' in data:
                edge_info = data['info']
                edges_dict.append({
                    'from': u,
                    'to': v,
                    'action': edge_info.action
                })
        
        return {
            'nodes': nodes_dict,
            'edges': edges_dict,
            'root_node_id': self.root_node.id if self.root_node else None,
            'statistics': self.get_search_statistics()
        }
    

    # ========== Visualization methods ==========

    def visualize_graph(
        self,
        highlight_path: Optional[List[MoleculeNode]] = None,
        highlight_actions: Optional[List[str]] = None,
        show_uct: bool = True,
        show_scores: bool = True,
        figsize: Tuple[int, int] = (16, 12),
        node_size: int = 2000,
        font_size: int = 10,
        # title: str = "Molecular Optimization Graph"
    ) -> plt.Figure:
        """
        Visualize the molecular optimization graph.
        
        Args:
            highlight_path: Path nodes to highlight.
            highlight_actions: Path actions paired with highlight_path.
            show_uct: Whether to display UCT values in node labels.
            show_scores: Whether to display scores in node labels.
            figsize: Figure size.
            node_size: Node size.
            font_size: Font size.
            title: Figure title.
            
        Returns:
            Matplotlib figure object.
        """
        # TODO: Include task requirements in the title and save path.
        title = f"{self.root_node.smiles}_iter{self.iteration_count-1}"
        if not self.graph.nodes():
            print("Graph is empty!")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Graph is empty", ha='center', va='center', fontsize=14)
            ax.set_title(title)
            return fig
        
        # Create the layout.
        try:
            # Use Graphviz dot layout (requires pygraphviz).
            pos = graphviz_layout(self.graph, prog='dot')
        except:
            # Fall back to spring layout if Graphviz is unavailable.
            print("Graphviz not available, using spring layout...")
            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # Create the figure.
        fig, ax = plt.subplots(figsize=figsize)
        
        # Encode node colors by agent type.
        node_colors = []
        node_labels = {}
        
        # Define the agent-type color map.
        agent_color_map = {
            'human_insert': 'blue',      # User-inserted root node.
            'creative_molecule_explorer': 'orange',           # Creative agent.
            'rational_medicinal_designer': 'green',            # Rational agent.
            # Use the default color for other cases.
            'default': 'lightgray'
        }
        
        for node_id in self.graph.nodes():
            node = self.get_node_by_id(node_id)
            if not node:
                continue
                
        # Determine node color from its agent type.
            if node == self.root_node:
                color = 'blue'  # Keep the root blue.
            elif node.unreachable:
                color = 'black'  # Keep unreachable nodes black.
            elif highlight_path and node in highlight_path:
                color = 'cyan'  # Keep the highlighted path cyan.
            else:
                # Choose the color by agent type.
                agent_type = getattr(node, 'agent_type', None)
                color = agent_color_map.get(agent_type, agent_color_map['default'])
            
            node_colors.append(color)
            
            # Create node labels.
            label_parts = []
            
            # Show the first 20 SMILES characters.
            smiles_short = node.smiles[:20] + "..." if len(node.smiles) > 20 else node.smiles
            label_parts.append(f"{node.id}\n{smiles_short}")
            
            if show_uct:
                label_parts.append(f"UCT: {node.uct_value:.3f}")
            
            if show_scores:
                label_parts.append(f"Intr: {node.intrinsic_score:.3f}")
                label_parts.append(f"Value: {node.average_reward():.3f}")
            
            label_parts.append(f"Visits: {node.visit_count}")
            
            # Add agent-type information to the label.
            if node.agent_type:
                label_parts.append(f"Agent: {node.agent_type}")
            
            node_labels[node_id] = "\n".join(label_parts)
        
        # Draw nodes.
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=node_size,
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges.
        edge_colors = []
        edge_widths = []
        
        for u, v in self.graph.edges():
            # Get edge information.
            edge_data = self.graph.get_edge_data(u, v)
            
            # Determine edge color and width.
            if highlight_path and highlight_actions:
                # Check whether this edge is on the highlighted path.
                in_path = False
                for i in range(len(highlight_path) - 1):
                    if (highlight_path[i].id == u and highlight_path[i + 1].id == v) or \
                    (highlight_path[i].id == v and highlight_path[i + 1].id == u):
                        in_path = True
                        break
                
                if in_path:
                    edge_colors.append('red')
                    edge_widths.append(3.0)
                else:
                    edge_colors.append('gray')
                    edge_widths.append(1.0)
            else:
                edge_colors.append('gray')
                edge_widths.append(1.0)
        
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.7,
            arrows=True,
            arrowsize=15,
            ax=ax
        )
        
        # Draw node labels.
        nx.draw_networkx_labels(
            self.graph, pos,
            labels=node_labels,
            font_size=font_size,
            font_family='monospace',
            ax=ax
        )
        
        # Draw edge labels (actions).
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            if 'info' in data and data['info'].action:
                edge_labels[(u, v)] = data['info'].action[:15]  # Truncate action text.
        
        if edge_labels:
            nx.draw_networkx_edge_labels(
                self.graph, pos,
                edge_labels=edge_labels,
                font_size=font_size - 2,
                ax=ax
            )
        
        # Add a legend.
        from matplotlib.patches import Patch
        
        legend_elements = [
            Patch(facecolor='blue', alpha=0.8, label='Root Node (Human)'),
            Patch(facecolor='orange', alpha=0.8, label='creative_molecule_explorer'),
            Patch(facecolor='green', alpha=0.8, label='rational_medicinal_designer'),
            Patch(facecolor='cyan', alpha=0.8, label='Selected Path'),
            Patch(facecolor='black', alpha=0.8, label='Unreachable'),
            Patch(facecolor='lightgray', alpha=0.8, label='Unknown Agent')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        # Set the title and axes.
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Adjust the layout.
        plt.tight_layout()

        # Save the image.
        output_path = _resolve_generated_file_path(title, MCG_RUNTIME_OUTPUT_DIR)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig




    def visualize_search_path(
        self,
        path_nodes: List[MoleculeNode],
        path_actions: List[str],
        figsize: Tuple[int, int] = (14, 8),
        title: str = "Selected Search Path"
    ) -> plt.Figure:
        """
        Visualize the selected search path.
        
        Args:
            path_nodes: Nodes along the path.
            path_actions: Actions along the path.
            figsize: Figure size.
            title: Figure title.
            
        Returns:
            Matplotlib figure object.
        """
        if not path_nodes:
            print("No path to visualize!")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No path available", ha='center', va='center', fontsize=14)
            ax.set_title(title)
            return fig
        
        # Create subplots.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Subplot 1: path visualization.
        # Create the path subplot.
        path_subgraph = nx.DiGraph()
        pos_path = {}
        
        for i, node in enumerate(path_nodes):
            path_subgraph.add_node(node.id, info=node)
            pos_path[node.id] = (i, 0)
            
            if i > 0:
                path_subgraph.add_edge(path_nodes[i-1].id, node.id, 
                                    action=path_actions[i-1] if i-1 < len(path_actions) else "Unknown")
        
        # Node colors.
        node_colors = ['blue' if i == 0 else 'cyan' for i in range(len(path_nodes))]
        
        nx.draw_networkx_nodes(
            path_subgraph, pos_path,
            node_color=node_colors,
            node_size=1500,
            alpha=0.8,
            ax=ax1
        )
        
        nx.draw_networkx_edges(
            path_subgraph, pos_path,
            edge_color='red',
            width=2.0,
            alpha=0.8,
            arrows=True,
            arrowsize=20,
            ax=ax1
        )
        
        # Node labels.
        node_labels = {}
        for i, node_id in enumerate(path_subgraph.nodes()):
            node = self.get_node_by_id(node_id)
            if node:
                smiles_short = node.smiles[:15] + "..." if len(node.smiles) > 15 else node.smiles
                node_labels[node_id] = f"{node_id}\n{smiles_short}\nIntr: {node.intrinsic_score:.3f}\nVisits: {node.visit_count}"
        
        nx.draw_networkx_labels(
            path_subgraph, pos_path,
            labels=node_labels,
            font_size=9,
            font_family='monospace',
            ax=ax1
        )
        
        # Edge labels.
        edge_labels = {}
        for u, v, data in path_subgraph.edges(data=True):
            if 'action' in data:
                edge_labels[(u, v)] = data['action'][:10]
        
        nx.draw_networkx_edge_labels(
            path_subgraph, pos_path,
            edge_labels=edge_labels,
            font_size=8,
            ax=ax1
        )
        
        ax1.set_title(f"Search Path (Length: {len(path_nodes)-1})", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Subplot 2: path-score statistics.
        # Extract score data.
        depths = list(range(len(path_nodes)))
        intrinsic_scores = [node.intrinsic_score for node in path_nodes]
        uct_values = [node.uct_value for node in path_nodes]
        visit_counts = [node.visit_count for node in path_nodes]
        
        # Draw the line chart.
        ax2.plot(depths, intrinsic_scores, 'b-o', label='Intrinsic Score', linewidth=2, markersize=8)
        ax2.plot(depths, uct_values, 'r-s', label='UCT Value', linewidth=2, markersize=8)
        ax2.set_xlabel('Depth from Root', fontsize=11)
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Path Scores Evolution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add a second y-axis for visit counts.
        ax2_twin = ax2.twinx()
        ax2_twin.bar(depths, visit_counts, alpha=0.3, color='green', label='Visit Count')
        ax2_twin.set_ylabel('Visit Count', fontsize=11)
        ax2_twin.legend(loc='upper right')
        
        # Set the title.
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


    def save_graph_visualization(
        self,
        filename: str,
        highlight_path: Optional[List[MoleculeNode]] = None,
        highlight_actions: Optional[List[str]] = None,
        dpi: int = 300,
        **kwargs
    ) -> None:
        """
        Save the graph visualization to a file.
        
        Args:
            filename: Output filename.
            highlight_path: Path to highlight.
            highlight_actions: Actions along the path.
            dpi: Image resolution.
            **kwargs: Additional arguments forwarded to visualize_graph.
        """
        fig = self.visualize_graph(
            highlight_path=highlight_path,
            highlight_actions=highlight_actions,
            **kwargs
        )
        
        output_path = _resolve_generated_file_path(filename, MCG_RUNTIME_OUTPUT_DIR)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Graph visualization saved to {output_path}")


    def get_search_statistics(self) -> Dict:
        """
        Return search statistics.
        
        Returns:
            Statistics dictionary.
        """
        all_nodes = self.get_all_nodes()
        
        if not all_nodes:
            return {
                'total_nodes': 0,
                'total_edges': 0,
                'reachable_nodes': 0
            }
        
        # Count reachable nodes.
        reachable_count = 0
        if self.root_node:
            for node in all_nodes:
                if node == self.root_node or nx.has_path(self.graph, self.root_node.id, node.id):
                    reachable_count += 1
        
        # Compute score statistics.
        intrinsic_scores = [node.intrinsic_score for node in all_nodes]
        uct_values = [node.uct_value for node in all_nodes]
        visit_counts = [node.visit_count for node in all_nodes]
        
        # Get the best node.
        best_intrinsic_node = max(all_nodes, key=lambda x: x.intrinsic_score) if all_nodes else None
        best_uct_node = max(all_nodes, key=lambda x: x.uct_value) if all_nodes else None
        
        return {
            'total_nodes': len(all_nodes),
            'total_edges': self.graph.number_of_edges(),
            'reachable_nodes': reachable_count,
            'average_intrinsic_score': np.mean(intrinsic_scores) if intrinsic_scores else 0,
            'max_intrinsic_score': max(intrinsic_scores) if intrinsic_scores else 0,
            'average_uct_value': np.mean(uct_values) if uct_values else 0,
            'max_uct_value': max(uct_values) if uct_values else 0,
            'average_visit_count': np.mean(visit_counts) if visit_counts else 0,
            'max_visit_count': max(visit_counts) if visit_counts else 0,
            'best_intrinsic_node': best_intrinsic_node.id if best_intrinsic_node else None,
            'best_uct_node': best_uct_node.id if best_uct_node else None,
            'iteration_count': self.iteration_count,
            'total_simulations': self.total_simulations
        }
    

    def save_nodes_to_csv(self, filename: str = 'graph_nodes.csv', include_root: bool = False, tmp: bool = False):
        """
        Save information about graph nodes to a CSV file.
        
        Args:
            filename: CSV filename; defaults to 'graph_nodes.csv'.
            include_root: Whether to include the root; defaults to False.
        
        Returns:
            bool: Whether saving succeeded.
        """

        try:
            output_filename = self._resolve_output_filename(filename, tmp=tmp)
            output_path = _resolve_generated_file_path(output_filename, MCG_RUNTIME_OUTPUT_DIR)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            all_nodes = self.get_all_nodes()
            
            if not all_nodes:
                print("Warning: Graph has no nodes to save")
                return False
            
            # Filter nodes, excluding the root if requested.
            nodes_to_save = []
            for node in all_nodes:
                if not include_root and self.root_node and node.id == self.root_node.id:
                    continue
                nodes_to_save.append(node)
            
            if not nodes_to_save:
                print("Warning: No nodes to save after filtering")
                return False
            
            # Prepare the CSV file.
            # filename = 
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = self._get_csv_fieldnames()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write each node's data.
                for node in nodes_to_save:
                    writer.writerow(self._node_to_csv_row(node))

            print(f"✓ Successfully saved {len(nodes_to_save)} nodes to {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error saving nodes to CSV: {e}")
            return False
    

def quick_save(graph: 'MolecularGraph', filepath: str = 'molecular_graph.pkl') -> bool:
    """
    Save a MolecularGraph object with pickle.
    
    Args:
        graph: MolecularGraph object.
        filepath: Save path.
    
    Returns:
        bool: Whether saving succeeded.
    """
    try:
        output_path = _resolve_generated_file_path(filepath, MCG_RUNTIME_OUTPUT_DIR)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(graph, f)
        print(f"✓ Graph quickly saved to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error quick saving graph: {e}")
        return False


def quick_load(filepath: str = 'molecular_graph.pkl') -> Optional['MolecularGraph']:
    """
    Load a MolecularGraph object from a pickle file.
    
    Args:
        filepath: File path.
    
    Returns:
        MolecularGraph object.
    """
    try:
        input_path = _resolve_generated_file_path(filepath, MCG_RUNTIME_OUTPUT_DIR)
        with open(input_path, 'rb') as f:
            graph = pickle.load(f)
        print(f"✓ Graph quickly loaded from {input_path}")
        print(f"  Nodes: {len(graph.get_all_nodes())}, Edges: {graph.graph.number_of_edges()}")
        return graph
    except Exception as e:
        print(f"✗ Error quick loading graph: {e}")
        return None
