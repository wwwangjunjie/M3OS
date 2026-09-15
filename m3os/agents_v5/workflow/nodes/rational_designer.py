"""Node: Rational Designer Execution

This node invokes the Rational Medicinal Designer agent to generate
optimized molecules using case-based reasoning and medicinal chemistry principles.
"""

from typing import Any, Dict
import time

from m3os.agents_v5.core.state import MCGSState
from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.core.trace_utils import (
    safe_serialize,
    structured_field,
    structured_optimizations,
)
from m3os.agents_v5.agents.rational_designer import RationalMedicinalDesignerAgent
from m3os.agents_v5.prompts import RATIONAL_DESIGNER_USER_PROMPT
from m3os.agents_v5.services.admet_property_selection import build_selected_admet_prompt_block
from m3os.agents_v5.services.molecular_context import (
    without_explicit_molecular_auxiliary_fields,
)
from m3os.agents_v5.workflow.audit_recovery import audit_with_empty_result_recovery
from m3os.agents_v5.workflow.memory import format_agent_memory
from m3os.agents_v5.workflow.screening_context import format_screening_context_history


class RationalDesignerNode:
    """Node for executing the Rational Medicinal Designer agent.
    
    This node invokes the rational agent with the current molecular context
    and returns the generated optimized molecules.
    """
    
    def __init__(
        self,
        config: AgentConfig,
        rational_agent: RationalMedicinalDesignerAgent,
        auditor_agent: Any = None,
    ):
        """Initialize the node.
        
        Args:
            config: Agent configuration
            rational_agent: Pre-configured rational agent instance
        """
        self.config = config
        self.rational_agent = rational_agent
        self.auditor_agent = auditor_agent
    
    async def __call__(self, state: MCGSState) -> Dict[str, Any]:
        """Execute the node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary of updates to the state
        """
        print("\n=== Node: rational_designer ===")
        
        # Get current molecular info
        current_info = state["current_info_list"][-1] if state["current_info_list"] else {}
        shared = state.get("current_shared_analysis") or {}
        round_index = len(state.get("current_info_list") or [])
        mcgs_round = state.get("current_mcgs_round") or round_index
        display_round = max(1, int(mcgs_round or 1))
        if hasattr(self.rational_agent, "set_event_context"):
            self.rational_agent.set_event_context(
                agent_type="rational_medicinal_designer",
                round=display_round,
                display_round=display_round,
                round_index=round_index,
                mcgs_round=display_round,
                display_mcgs_round=display_round,
                initial_smiles=state.get("initial_smiles"),
                initial_protein_sequence=state.get("initial_protein_squence"),
                current_smiles=current_info.get("current_smiles"),
            )
        start = time.perf_counter()
        
        # Format the user prompt
        query = RATIONAL_DESIGNER_USER_PROMPT.format(
            initial_smiles=state["initial_smiles"],
            initial_iupac=state["initial_iupac"],
            initial_fragments=state["initial_fragments"],
            optimization_goal=state["optimization_goal"],
            current_smiles=current_info.get("current_smiles"),
            current_iupac=current_info.get("current_iupac"),
            current_fragments=current_info.get("current_fragments"),
            project_manager_brief=state.get("project_manager_brief", ""),
            current_selection_context=current_info.get("current_selection_context"),
            current_mcgs_round=display_round,
            initial_protein_squence=state.get("initial_protein_squence"),
            shared_analysis_summary=shared.get("shared_analysis_summary", ""),
            shared_keep_fragments=shared.get("shared_keep_fragments", []),
            shared_modifiable_fragments=shared.get("shared_modifiable_fragments", []),
            shared_risk_alerts=shared.get("shared_risk_alerts", []),
            shared_priority_directions=shared.get("shared_priority_directions", []),
            agent_memory=format_agent_memory(state.get("agent_memory"), "rational"),
            screening_context_history=format_screening_context_history(
                state.get("screening_context_history")
            ),
        )
        if not self.config.enable_molecular_auxiliary_context:
            query = without_explicit_molecular_auxiliary_fields(query)
        query += build_selected_admet_prompt_block(state)
        initial_interaction_evidence = (
            self.rational_agent.get_initial_interaction_evidence()
            if self.config.enable_molecular_auxiliary_context
            else ""
        )
        if initial_interaction_evidence:
            query += (
                "\n\n[INITIAL PROTEIN-LIGAND INTERACTION EVIDENCE]\n"
                f"{initial_interaction_evidence}\n"
                "Reuse this text for structural reasoning. Do not call Boltz or PLIP "
                "again and do not claim that unmodeled candidates preserve these contacts."
            )
        
        elapsed = round(time.perf_counter() - start, 3)
        try:
            result = await self.rational_agent.invoke(query)
            audit_result = await self._structured_generation_output(
                result,
                query=query,
                agent_type="rational_medicinal_designer",
            )
            source_agent_result = (
                audit_result.get("source_agent_result")
                if isinstance(audit_result.get("source_agent_result"), dict)
                else result
            )
            agent_output = audit_result["structured_response"]
            processed_molecules = self._process_output(agent_output, "rational_medicinal_designer")
            trace_payload = {
                "round_index": round_index,
                "mcgs_round": display_round,
                "display_mcgs_round": display_round,
                "node": "rational_designer",
                "agent_type": "rational_medicinal_designer",
                "query": query,
                "raw_agent_response": audit_result.get("raw_agent_response")
                or (source_agent_result.get("raw_response_text") if isinstance(source_agent_result, dict) else None),
                "audit_source_response": audit_result.get("audit_source_text"),
                "invocation_trace": source_agent_result.get("invocation_trace") if isinstance(source_agent_result, dict) else None,
                "auditor_trace": audit_result.get("auditor_trace"),
                "audit_finalization_trace": audit_result.get("audit_finalization_trace"),
                "audit_recovery_trace": audit_result.get("audit_recovery_trace"),
                "audit_warnings": audit_result.get("audit_warnings", []),
                "smiles_validation_gate": audit_result.get("smiles_validation_gate"),
                "processed_molecules": processed_molecules,
            }
            llm_usage = {
                "agent": source_agent_result.get("llm_usage") if isinstance(source_agent_result, dict) else None,
                "finalizer": audit_result.get("finalization_llm_usage"),
                "auditor": audit_result.get("llm_usage"),
            }
            error = None
        except Exception as exc:
            elapsed = round(time.perf_counter() - start, 3)
            processed_molecules = []
            error = str(exc)
            trace_payload = {
                "round_index": round_index,
                "node": "rational_designer",
                "agent_type": "rational_medicinal_designer",
                "query": query,
                "error": error,
                "processed_molecules": processed_molecules,
            }
            llm_usage = None
            print(f"Rational designer failed, continuing with empty results: {error}")

        trace_payload = {
            **trace_payload,
            "failed": error is not None,
        }

        print(processed_molecules)
        print("\n===[END] Node: rational_designer ===")

        return {
            "optimized_molecules_rational": processed_molecules,
            "rational_agent_traces": [trace_payload],
            "runtime_metrics": [
                {
                    "node": "rational_designer",
                    "elapsed_sec": elapsed,
                    "llm_usage": llm_usage,
                    "error": error,
                }
            ],
        }

    async def _structured_generation_output(
        self,
        result: Any,
        *,
        query: str,
        agent_type: str,
    ) -> Dict[str, Any]:
        if isinstance(result, dict) and result.get("structured_response") is not None:
            return {
                "structured_response": result["structured_response"],
                "audit_warnings": [],
                "auditor_trace": None,
                "llm_usage": None,
            }
        if self.auditor_agent is None:
            raise ValueError("Rational designer returned free-form output but no Auditor agent is configured.")
        audit_result = await audit_with_empty_result_recovery(
            agent=self.rational_agent,
            auditor_agent=self.auditor_agent,
            result=result,
            audit_kind="molecule_optimizations",
            source_agent_type=agent_type,
            audit_method_name="audit_molecule_optimizations",
            rerun_query=query,
        )
        if audit_result.get("audit_warnings"):
            print("[Rational Auditor warnings]:", safe_serialize(audit_result["audit_warnings"]))
        return audit_result
    
    def _process_output(
        self,
        agent_output: Any,
        agent_type: str
    ) -> list:
        """Process agent output into standardized format.
        
        Args:
            agent_output: Raw agent output
            agent_type: Type of agent that produced the output
            
        Returns:
            List of processed molecule dictionaries
        """
        molecules_list = structured_optimizations(agent_output)
        processed = []
        
        for molecule in molecules_list:
            processed.append({
                "SMILES": structured_field(molecule, "smiles", ""),
                "modification_type": structured_field(molecule, "modification_type", ""),
                "rationale": structured_field(molecule, "rationale", ""),
                "confidence_score": structured_field(molecule, "confidence_score", 0.0),
                "agent_type": agent_type,
            })
        
        return processed
