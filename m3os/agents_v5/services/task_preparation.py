"""Task preparation service for molecular optimization sessions.

This module owns the non-MCGS work needed before an expansion step can run:
extracting user intent into a project-manager brief, resolving fragments/IUPAC
names, and freezing the task-level ADMET endpoint policy.
"""

from typing import Any, Dict, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.core.models import (
    ExtractInitialInfo,
)
from m3os.agents_v5.core.retry_utils import (
    invoke_with_retry,
    validate_structured_response,
)
from m3os.agents_v5.services.knowledge_context import KnowledgeContextService
from m3os.agents_v5.services.llm_factory import create_base_model
from m3os.agents_v5.services.mcp_client import MCPClientManager
from m3os.agents_v5.services.smiles_validation import validate_smiles
from m3os.agents_v5.services.admet_property_selection import (
    ADMETPropertySelectionService,
)
from m3os.agents_v5.prompts import (
    EXTRACT_INFO_SYSTEM_PROMPT,
)

class TaskPreparationService:
    """Prepare optimization state from a user request or conversation summary."""

    def __init__(
        self,
        config: AgentConfig,
        mcp_client: MCPClientManager,
        creative_agent: Any = None,
        rational_agent: Any = None,
        critic_agent: Any = None,
        knowledge_context: KnowledgeContextService | None = None,
    ):
        self.config = config
        self.mcp_client = mcp_client
        self.creative_agent = creative_agent
        self.rational_agent = rational_agent
        self.critic_agent = critic_agent
        self.knowledge_context = knowledge_context or KnowledgeContextService(config)
        set_mcp_client = getattr(self.knowledge_context, "set_mcp_client", None)
        if callable(set_mcp_client):
            set_mcp_client(mcp_client)

    async def prepare_initial_state(
        self,
        user_prompt: str,
        project_manager_brief: Optional[str] = None,
        additional_context: Optional[str] = None,
        run_initial_analysis: bool = True,
    ) -> Dict[str, Any]:
        """Return MCGSState updates for a newly prepared optimization task."""
        print("\n=== Service: task_preparation ===")

        extraction_prompt = self._build_extract_info_prompt(
            user_prompt=user_prompt,
            additional_context=additional_context,
        )
        extract_info = await self.extract_info(extraction_prompt)
        submitted_smiles = extract_info.smiles
        is_valid_smiles, canonical_smiles, smiles_error = validate_smiles(submitted_smiles)
        if not is_valid_smiles:
            raise ValueError(
                "Invalid SMILES. Please provide a valid molecular SMILES string."
                f" RDKit validation failed: {smiles_error}"
            )
        current_smiles = canonical_smiles or str(submitted_smiles or "").strip()
        current_smiles_cache_key = self._canonical_smiles_key(current_smiles)
        optimization_goal = extract_info.optimization_goal
        node_num_needed, iteration_num_needed, iteration_num_defaulted = (
            self._normalize_search_limits(extract_info)
        )
        prepared_project_manager_brief = self._build_project_manager_brief(
            extract_info=extract_info,
            user_prompt=user_prompt,
            project_manager_brief=project_manager_brief,
        )
        protein_sequence = extract_info.protein_sequence

        print(f"""Current SMILES: {current_smiles},
Optimization Goal: {optimization_goal},
Project Manager Brief: {prepared_project_manager_brief},
Protein Sequence: {protein_sequence},
Requested Candidates: {node_num_needed},
Requested Iterations: {iteration_num_needed}""")

        fragments, iupac = await self._molecular_auxiliary_facts(current_smiles)

        selected_admet = await ADMETPropertySelectionService(self.config).select_for_task(
            initial_smiles=current_smiles,
            initial_iupac=iupac,
            initial_fragments=fragments,
            optimization_goal=optimization_goal,
            project_manager_brief=prepared_project_manager_brief,
            protein_sequence=protein_sequence,
        )

        print("\n===[END] Service: task_preparation ===")

        shared_analysis_cache: Dict[str, Any] = {}

        return {
            "current_shared_analysis": {
                "shared_analysis_summary": "",
                "shared_keep_fragments": [],
                "shared_modifiable_fragments": [],
                "shared_risk_alerts": [],
                "shared_priority_directions": [],
            },
            "shared_analysis_cache": shared_analysis_cache,
            "molecule_facts_cache": {
                current_smiles_cache_key: {
                    "smiles": current_smiles,
                    "canonical_smiles": current_smiles_cache_key,
                    "iupac": iupac,
                    "fragments": fragments,
                }
            },
            "initial_smiles": current_smiles,
            "initial_iupac": iupac,
            "initial_fragments": fragments,
            "optimization_goal": optimization_goal,
            "project_manager_brief": prepared_project_manager_brief,
            **selected_admet,
            "current_info_list": [
                {
                    "current_smiles": current_smiles,
                    "current_iupac": iupac,
                    "current_fragments": fragments,
                    "current_selection_context": "Initial molecule, no selection context yet.",
                    "current_select_reason": "Initial molecule, no selection needed.",
                }
            ],
            "optimized_molecules": [current_smiles],
            "optimized_molecules_creative": [],
            "optimized_molecules_rational": [],
            "node_num_needed": node_num_needed,
            "iteration_num_needed": iteration_num_needed,
            "iteration_num_defaulted": iteration_num_defaulted,
            "agent_memory": {"rational": [], "creative": [], "critic": []},
            "screening_context_history": [],
            "initial_protein_squence": protein_sequence,
            "additional_context": additional_context,
            "runtime_metrics": [],
        }

    async def prepare_existing_graph_context(
        self,
        user_prompt: str,
        *,
        base_smiles: str,
        project_manager_brief: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return task-context updates for continuing from an existing graph node.

        Unlike prepare_initial_state, this intentionally does not return
        initial_* fields or graph fields. The caller preserves the existing MCGS
        root and graph topology while refreshing goal-level context.
        """
        print("\n=== Service: task_preparation existing graph context ===")

        extraction_prompt = self._build_extract_info_prompt(
            user_prompt=user_prompt,
            additional_context=additional_context,
        )
        extract_info = await self.extract_info(extraction_prompt)
        is_valid_smiles, canonical_smiles, smiles_error = validate_smiles(base_smiles)
        if not is_valid_smiles:
            raise ValueError(
                "Invalid base SMILES. Please provide a valid molecular SMILES string."
                f" RDKit validation failed: {smiles_error}"
            )
        current_smiles = canonical_smiles or str(base_smiles or "").strip()
        current_smiles_cache_key = self._canonical_smiles_key(current_smiles)
        optimization_goal = extract_info.optimization_goal
        node_num_needed, iteration_num_needed, iteration_num_defaulted = (
            self._normalize_search_limits(extract_info)
        )
        prepared_project_manager_brief = self._build_project_manager_brief(
            extract_info=extract_info,
            user_prompt=user_prompt,
            project_manager_brief=project_manager_brief,
        )
        protein_sequence = extract_info.protein_sequence

        print(f"""Continuation Base SMILES: {current_smiles},
Optimization Goal: {optimization_goal},
Project Manager Brief: {prepared_project_manager_brief},
Protein Sequence: {protein_sequence},
Requested Candidates: {node_num_needed},
Requested Iterations: {iteration_num_needed}""")

        fragments, iupac = await self._molecular_auxiliary_facts(current_smiles)

        selected_admet = await ADMETPropertySelectionService(self.config).select_for_task(
            initial_smiles=current_smiles,
            initial_iupac=iupac,
            initial_fragments=fragments,
            optimization_goal=optimization_goal,
            project_manager_brief=prepared_project_manager_brief,
            protein_sequence=protein_sequence,
        )

        print("\n===[END] Service: task_preparation existing graph context ===")

        return {
            "user_prompt": user_prompt,
            "additional_context": additional_context,
            "current_shared_analysis": {
                "shared_analysis_summary": "",
                "shared_keep_fragments": [],
                "shared_modifiable_fragments": [],
                "shared_risk_alerts": [],
                "shared_priority_directions": [],
            },
            "shared_analysis_cache": {},
            "molecule_facts_cache": {
                current_smiles_cache_key: {
                    "smiles": current_smiles,
                    "canonical_smiles": current_smiles_cache_key,
                    "iupac": iupac,
                    "fragments": fragments,
                }
            },
            "optimization_goal": optimization_goal,
            "project_manager_brief": prepared_project_manager_brief,
            **selected_admet,
            "optimized_molecules": [],
            "optimized_molecules_creative": [],
            "optimized_molecules_rational": [],
            "node_num_needed": node_num_needed,
            "iteration_num_needed": iteration_num_needed,
            "iteration_num_defaulted": iteration_num_defaulted,
            "initial_protein_squence": protein_sequence,
            "invalid_molecules": [],
            "invalid_molecules_filtered": 0,
            "expansion_error": "",
        }

    @staticmethod
    def _normalize_search_limits(
        extract_info: ExtractInitialInfo,
    ) -> tuple[int, Optional[int], bool]:
        candidate_count = TaskPreparationService._positive_int_or_none(
            extract_info.node_num_needed
        )
        round_count = TaskPreparationService._positive_int_or_none(
            extract_info.iteration_num_needed
        )
        round_count_defaulted = False

        if candidate_count is None and round_count is None:
            round_count = 1
            round_count_defaulted = True

        return candidate_count or 10, round_count, round_count_defaulted

    @staticmethod
    def _positive_int_or_none(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return max(1, parsed) if parsed > 0 else None

    @staticmethod
    def _build_project_manager_brief(
        *,
        extract_info: ExtractInitialInfo,
        user_prompt: str,
        project_manager_brief: Optional[str] = None,
    ) -> str:
        brief = str(project_manager_brief or "").strip()
        if brief:
            return brief
        brief = str(extract_info.project_manager_brief or "").strip()
        if brief:
            return brief
        parts = [
            f"Optimization goal: {extract_info.optimization_goal}",
            "Use the user's latest natural-language request as the project manager instruction.",
            f"Original request: {str(user_prompt or '').strip()}",
        ]
        if extract_info.iteration_num_needed:
            parts.append(f"Requested MCGS expansion rounds: {extract_info.iteration_num_needed}")
        if extract_info.node_num_needed:
            parts.append(f"Requested optimized candidate molecules: {extract_info.node_num_needed}")
        return "\n".join(part for part in parts if str(part).strip())

    @staticmethod
    def _canonical_smiles_key(smiles: Any) -> str:
        ok, canonical, _error = validate_smiles(smiles)
        return canonical if ok and canonical else str(smiles or "").strip()

    @classmethod
    def _build_extract_info_prompt(
        cls,
        *,
        user_prompt: str,
        additional_context: Optional[str] = None,
    ) -> str:
        structure_context = cls._extract_uploaded_structure_context(additional_context)
        if not structure_context:
            return user_prompt
        return (
            f"{str(user_prompt or '').strip()}\n\n"
            "[UPLOADED STRUCTURE FILE CONTEXT FOR EXTRACTION]\n"
            "The user uploaded machine-readable structure files. If the latest user "
            "request does not explicitly override these values, extract the primary "
            "SDF SMILES as `smiles` and the PDB FASTA/protein sequence as "
            "`protein_sequence`.\n"
            f"{structure_context}\n"
            "[END UPLOADED STRUCTURE FILE CONTEXT FOR EXTRACTION]"
        )

    @staticmethod
    def _extract_uploaded_structure_context(additional_context: Optional[str]) -> Optional[str]:
        text = str(additional_context or "")
        start_marker = "[USER-UPLOADED STRUCTURE FILE CONTEXT]"
        end_marker = "[END USER-UPLOADED STRUCTURE FILE CONTEXT]"
        blocks = []
        search_from = 0
        while True:
            start = text.find(start_marker, search_from)
            if start == -1:
                break
            end = text.find(end_marker, start)
            if end == -1 or end <= start:
                break
            end += len(end_marker)
            blocks.append(text[start:end].strip())
            search_from = end
        if not blocks:
            return None
        return "\n\n".join(blocks)

    async def extract_info(self, user_prompt: str) -> ExtractInitialInfo:
        return await self.extract_info_only(user_prompt)

    async def extract_info_only(self, user_prompt: str) -> ExtractInitialInfo:
        llm = create_base_model(self.config.llm)
        structured_llm = llm.with_structured_output(ExtractInitialInfo, include_raw=True)

        messages = [
            SystemMessage(content=EXTRACT_INFO_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        async def _invoke():
            return await structured_llm.ainvoke(messages)

        response = await invoke_with_retry(
            _invoke,
            max_retries=3,
            base_delay=1.0,
            validate_fn=validate_structured_response,
            operation_name="Extract initial info LLM call",
        )
        print("[EXTRACT INFO TOKEN]:", response["raw"].usage_metadata)
        return response["parsed"]

    async def get_fragments(self, smiles: str) -> Any:
        try:
            fragments = await self.mcp_client.invoke_tool(
                tool_name="smiles_to_fragments_str",
                params={"smiles_list": [smiles]},
            )
            print(f"Fragments: {fragments}")
            return fragments
        except Exception as exc:
            print(f"Error fragmenting molecule: {exc}")
            try:
                from m3os.agents_v5.tools.fragments import smiles2fragments_str

                fragments = smiles2fragments_str([smiles])
                print(f"Fragments from local fallback: {fragments}")
                return fragments
            except Exception as fallback_exc:
                print(f"Error fragmenting molecule with local fallback: {fallback_exc}")
                return "None"

    async def _molecular_auxiliary_facts(self, smiles: str) -> tuple[Any, Any]:
        if not self.config.enable_molecular_auxiliary_context:
            print("Molecular auxiliary context disabled: skipping fragments and IUPAC lookup.")
            return "", ""
        fragments = await self.get_fragments(smiles)
        iupac = await self.get_iupac(smiles)
        return fragments, iupac

    async def get_iupac(self, smiles: str) -> Any:
        try:
            iupac = await self.mcp_client.invoke_tool(
                tool_name="generate_iupac_name",
                params={"smiles_list": [smiles]},
                server_hint="mcp_server_iupac_gen",
            )
            print(f"IUPAC: {iupac}")
            return iupac
        except Exception as exc:
            print(f"Error generating IUPAC: {exc}")
            return "None"
