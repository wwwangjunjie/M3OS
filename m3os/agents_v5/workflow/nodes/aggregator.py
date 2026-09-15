"""Node: Molecules Aggregator

This node combines the results from both the Creative and Rational agents
into a single list of optimized molecules.
"""

from typing import Any, Dict, List

from m3os.agents_v5.core.state import MCGSState
from m3os.agents_v5.core.trace_utils import format_trace_block
from m3os.agents_v5.services.smiles_validation import compact_invalid_molecules, split_valid_molecules
from m3os.agents_v5.workflow.memory import append_agent_memory, generation_memory_items


class MoleculesAggregatorNode:
    """Node for aggregating molecules from multiple agents.
    
    This node combines the optimized molecules from both the Creative
    Molecule Explorer and the Rational Medicinal Designer into a single
    list for evaluation by the Critic.
    """
    
    def __call__(self, state: MCGSState) -> Dict[str, Any]:
        """Execute the node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary of updates to the state
        """
        print("\n=== Node: molecules_aggregator ===")
        
        # Get molecules from both agents
        creative_molecules = state.get("optimized_molecules_creative", [])
        rational_molecules = state.get("optimized_molecules_rational", [])
        creative_trace = (state.get("creative_agent_traces") or [None])[-1]
        rational_trace = (state.get("rational_agent_traces") or [None])[-1]
        
        # Simple merge, then drop invalid/duplicate SMILES before critic evaluation.
        merged_molecules = creative_molecules + rational_molecules
        optimized_molecules, invalid_molecules = split_valid_molecules(merged_molecules)
        trace_round = {
            "round_index": max(
                creative_trace.get("round_index", 0) if creative_trace else 0,
                rational_trace.get("round_index", 0) if rational_trace else 0,
            ),
            "creative": creative_trace,
            "rational": rational_trace,
            "invalid_molecules_filtered": len(invalid_molecules),
            "invalid_molecules": compact_invalid_molecules(invalid_molecules),
        }
        agent_memory = self._updated_agent_memory(
            state.get("agent_memory"),
            creative_trace=creative_trace,
            rational_trace=rational_trace,
        )
        
        print(f"Combined {len(creative_molecules)} creative and {len(rational_molecules)} rational molecules")
        print(f"Total valid: {len(optimized_molecules)} molecules")
        if invalid_molecules:
            print(f"Filtered {len(invalid_molecules)} invalid or duplicate molecules before critic evaluation")
        if creative_trace:
            print(format_trace_block("creative_molecule_explorer", creative_trace))
        if rational_trace:
            print(format_trace_block("rational_medicinal_designer", rational_trace))
        
        print("\n===[END] Node: molecules_aggregator ===")
        
        return {
            "optimized_molecules": optimized_molecules,
            "subagent_trace_rounds": [trace_round],
            "agent_memory": agent_memory,
            "invalid_molecules_filtered": len(invalid_molecules),
            "invalid_molecules": compact_invalid_molecules(invalid_molecules),
        }

    def _updated_agent_memory(
        self,
        memory: Any,
        *,
        creative_trace: Dict[str, Any] | None,
        rational_trace: Dict[str, Any] | None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        updated = append_agent_memory(
            memory,
            "creative",
            generation_memory_items(
                (creative_trace or {}).get("processed_molecules") or [],
                round_index=(creative_trace or {}).get("round_index"),
                source_agent="creative_molecule_explorer",
            ),
        )
        updated = append_agent_memory(
            updated,
            "rational",
            generation_memory_items(
                (rational_trace or {}).get("processed_molecules") or [],
                round_index=(rational_trace or {}).get("round_index"),
                source_agent="rational_medicinal_designer",
            ),
        )
        return updated
