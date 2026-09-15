"""Node 2: Initialize Graph

This node evaluates the initial molecule and adds it as the root node
of the Monte Carlo Graph Search (MCGS) structure.
"""

from typing import Any, Dict

from m3os.agents_v5.core.state import MCGSState
from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.core.trace_utils import (
    safe_serialize,
    structured_field,
    structured_optimizations,
)
from m3os.agents_v5.services.admet_property_selection import (
    build_selected_admet_prompt_block,
)
from m3os.agents_v5.services.molecule_visual_context import (
    build_topology_visual_context,
    contexts_to_metadata,
    text_block,
)
from m3os.agents_v5.services.molecular_context import (
    without_explicit_molecular_auxiliary_fields,
)
from m3os.agents_v5.agents.critic import CriticAgent
from m3os.agents_v5.services.mcp_client import MCPClientManager
from m3os.agents_v5.tools.mcgs_tools import (
    mcgs_add_root_node,
)
from m3os.agents_v5.workflow.audit_recovery import audit_with_empty_result_recovery
from m3os.agents_v5.tools.monte_carlo_graph.graph_for_molecular_optimization_v3_4 import MolecularGraph


class InitGraphNode:
    """Node for initializing the MCGS graph.
    
    This node:
    1. Initializes the Critic agent
    2. Evaluates the initial molecule
    3. Creates the molecular graph with the initial molecule as root
    4. Visualizes the initial graph
    """
    
    def __init__(
        self,
        config: AgentConfig,
        mcp_client: MCPClientManager,
        critic_agent: CriticAgent,
        auditor_agent: Any = None,
    ):
        """Initialize the node.
        
        Args:
            config: Agent configuration
            mcp_client: MCP client manager
            critic_agent: Pre-configured critic agent instance
        """
        self.config = config
        self.mcp_client = mcp_client
        self.critic_agent = critic_agent
        self.auditor_agent = auditor_agent
    
    async def __call__(self, state: MCGSState) -> Dict[str, Any]:
        """Execute the node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary of updates to the state
        """
        print("\n=== Node: Init Graph ===")
        
        print("\n=== Call Critic ===")
        
        # Initialize MCGS graph
        mol_graph = MolecularGraph(seed=2026)
        
        # Evaluate initial molecule with Critic
        initial_eval = await self._evaluate_initial_molecule(state)
        
        initial_eval_item = self._first_initial_evaluation(initial_eval)
        raw_properties = self._eval_field(initial_eval_item, "properties", {})
        properties = dict(raw_properties) if isinstance(raw_properties, dict) else {}

        # Add root node to graph
        mol_graph = mcgs_add_root_node(
            mol_graph=mol_graph,
            root_smiles=state["initial_smiles"],
            intrinsic_score=float(self._eval_field(initial_eval_item, "score", 0.0) or 0.0),
            properties=properties,
        )
        
        print("\n===[END] Node: Init Graph ===")
        
        return {
            "mcgs_graph": mol_graph
        }
    
    async def _evaluate_initial_molecule(self, state: MCGSState) -> Any:
        """Evaluate the initial molecule using the Critic agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Critic evaluation result
        """
        from m3os.agents_v5.prompts import CRITIC_USER_PROMPT
        
        # Format the user prompt for critic
        query = CRITIC_USER_PROMPT.format(
            initial_smiles=state["initial_smiles"],
            initial_iupac=state["initial_iupac"],
            initial_fragments=state["initial_fragments"],
            optimization_goal=state["optimization_goal"],
            project_manager_brief=state.get("project_manager_brief", ""),
            current_smiles=None,
            current_iupac=None,
            current_fragments=None,
            current_selection_context="Initial root molecule.",
            optimized_molecules=f"[{state['initial_smiles']}]",
            initial_protein_squence=state.get("initial_protein_squence"),
            shared_analysis_summary="",
            shared_keep_fragments=[],
            shared_modifiable_fragments=[],
            shared_risk_alerts=[],
            shared_priority_directions=[],
            agent_memory="No prior audited memory for this role.",
        )
        if not self.config.enable_molecular_auxiliary_context:
            query = without_explicit_molecular_auxiliary_fields(query)
        query += build_selected_admet_prompt_block(state)
        
        # Add special instruction for initial evaluation
        # query += "\n\nFor this task, you only need to analyze the initial molecule. There is no optimized molecule, so no optimization is required!!!"
        # query += "\n\nReturn ONE evaluation for the initial molecule with its current properties. DO NOT return an empty list."
        query += (
            "\n\nFor this task, you only need to analyze the initial molecule. "
            "Give a natural-language evaluation of the initial molecule. "
            "Explicitly include the initial molecule's smiles, action, rationale, score, properties, and agent_type "
            "in that evaluation so the workflow finalization step can summarize one initial molecule evaluation item.\n\n"
            "[Note]: Please keep all of your reasoning concise and clear. Avoid all unnecessary wording; include only the essentials!!!"
        )
        
        if hasattr(self.critic_agent, "set_event_context"):
            self.critic_agent.set_event_context(
                agent_type="critic",
                round="initial",
                display_round="initial",
                selected_admet_preference_json=state.get("selected_admet_preference_json"),
                selected_admet_properties=state.get("selected_admet_properties"),
                selected_admet_property_details=state.get("selected_admet_property_details"),
                initial_protein_sequence=state.get("initial_protein_squence"),
                protein_fasta_path=state.get("protein_fasta_path"),
                initial_smiles=state.get("initial_smiles"),
                current_smiles=state.get("initial_smiles"),
                leadopt_task_contract=state.get("leadopt_task_contract"),
            )

        # Invoke critic agent. The SMILES-only ablation must not render or send
        # topology context at all, including a nominal unavailable-image block.
        visual_context = None
        visual_invocation_failed = False
        if not self.config.enable_molecular_auxiliary_context:
            result = await self.critic_agent.invoke(query)
        else:
            visual_context = build_topology_visual_context(
                smiles=state["initial_smiles"],
                visual_id="INITIAL_MOLECULE",
                label="initial molecule",
                iupac=state.get("initial_iupac"),
                fragments=state.get("initial_fragments"),
                image_prefix="initial_molecule_topology",
            )
            visual_query = [
                {
                    "role": "user",
                    "content": [
                        text_block(
                            f"{query}\n\n"
                            "[CRITIC VISUAL ALIGNMENT RULES]\n"
                            "The following molecule visual context belongs to the initial/root molecule only. "
                            "Use the image together with the initial SMILES/IUPAC/fragments; do not infer a different molecule."
                        ),
                        *visual_context.content_blocks,
                    ],
                }
            ]
            try:
                result = await self.critic_agent.invoke(visual_query)
            except Exception as exc:
                visual_invocation_failed = True
                visual_context.metadata["included_in_model_call"] = False
                visual_context.metadata.setdefault(
                    "visual_context_error",
                    f"visual invocation failed: {type(exc).__name__}: {exc}",
                )
                print(
                    "[Initial Critic] Visual-context invocation failed; retrying text-only "
                    f"fallback: {type(exc).__name__}: {exc}"
                )
                result = await self.critic_agent.invoke(
                    f"{query}\n\n[VISUAL CONTEXT FALLBACK]\n"
                    "Initial molecule topology image was unavailable to the model; evaluate using SMILES/IUPAC/fragments."
                )

        print("[Critic Result]:", result)
        if isinstance(result, dict) and result.get("structured_response") is not None:
            result = dict(result)
            if visual_context is not None:
                result["visual_contexts"] = contexts_to_metadata([visual_context])
            if visual_invocation_failed:
                result["visual_context_fallback"] = True
            return result["structured_response"]
        if self.auditor_agent is None:
            raise ValueError("Initial critic evaluation returned free-form output but no Auditor agent is configured.")
        audit_result = await audit_with_empty_result_recovery(
            agent=self.critic_agent,
            auditor_agent=self.auditor_agent,
            result=result,
            audit_kind="critic_evaluations",
            source_agent_type="critic",
            audit_method_name="audit_critic_evaluations",
            rerun_query=query,
            extra_finalization_instruction=(
                "For this initial-root evaluation, include exactly one row for the initial molecule."
            ),
        )
        if audit_result.get("audit_warnings"):
            print("[Initial Critic Auditor warnings]:", safe_serialize(audit_result["audit_warnings"]))
        return audit_result["structured_response"]

    @staticmethod
    def _first_initial_evaluation(initial_eval: Any) -> Any:
        optimizations = structured_optimizations(initial_eval)
        if not isinstance(optimizations, list) or not optimizations:
            raise ValueError(
                "Critic returned no structured initial molecule evaluation. "
                "Expected top-level optimizations with one evaluated molecule."
            )
        return optimizations[0]

    @staticmethod
    def _eval_field(eval_res: Any, field: str, default: Any = None) -> Any:
        return structured_field(eval_res, field, default)
