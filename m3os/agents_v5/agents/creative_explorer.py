"""Creative Molecule Explorer agent.

This agent explores a broader chemical space through generative models,
using external REINVENT MCP tools for molecular generation and ADMET AI for filtering.
"""

import ast
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Mapping, Optional
import uuid

import pandas as pd

from langchain.tools import tool

from m3os.agents_v5.agents.base import BaseAgent
from m3os.agents_v5.core.config import AgentConfig, PROJECT_TMP_DIR
from m3os.agents_v5.services.mcp_client import (
    MCPClientManager,
    filter_tools_by_name,
)
from m3os.agents_v5.services.admet_property_selection import (
    canonical_admet_preference_json,
    enforce_frozen_admet_preference_json,
)
from m3os.agents_v5.prompts.system_loader import render_creative_system_prompt
from m3os.agents_v5.services.smiles_validation import validate_smiles


CREATIVE_DIRECT_MCP_TOOL_NAMES = {
    "prepare_smi_input",
    "prepare_scaffold",
    "validate_smiles_batch",
}

CREATIVE_WRAPPED_MCP_TOOL_NAMES = {
    "generate_similarity_constrained_mol2mol",
    "setup_generation_Mol2Mol_LinkInvent",
    "setup_libinvent",
    "run_generation",
    "admet_filter_by_admetai",
    "nesso_filter_by_cofolding",
    "screen_reinvent_candidates_by_constraints",
    "screen_reinvent_candidates_by_intersection",
}

ACTIVITY_GENERATION_SAMPLE_COUNT = 10000
INTERSECTION_GENERATION_SAMPLE_COUNTS = (10000,)
INTERSECTION_GENERATION_SAMPLE_COUNT = INTERSECTION_GENERATION_SAMPLE_COUNTS[0]
INTERSECTION_MAX_ATTEMPTS = len(INTERSECTION_GENERATION_SAMPLE_COUNTS)
MOL2MOL_CANDIDATE_COUNT = 10000
MOL2MOL_SAMPLES_PER_ROUND = 1000
MOL2MOL_MAX_ROUNDS = 10
MOL2MOL_DEFAULT_SIMILARITY_THRESHOLD = 0.7
ACTIVITY_FINALIST_COUNT = 10
DEFAULT_FINALIST_COUNT = 5
NESSO_AFFINITY_COLUMN = "Nesso_affinity_pred_value"
NESSO_PROBABILITY_COLUMN = "Nesso_affinity_probability_binary"
NESSO_AFFINITY_GATE_EPSILON = 1e-9
SCREENING_CONTEXT_NEAREST_COUNT = 10

SCREENING_CONTEXT_EXCLUDED_COLUMNS = {
    "Scaffold",
    "Input_SMILES",
    "Warheads",
    "R-groups",
    "Linker",
    "average_rank",
}

DISALLOWED_ADMET_CSV_PLACEHOLDERS = {
    "tmp/candidates.csv",
}


class CreativeMoleculeExplorerAgent(BaseAgent):
    """Agent for creative molecular exploration using generative models.
    
    This agent excels at exploring broader chemical space through:
    1. Diagnosing molecular defects and constructing optimization templates
    2. Using REINVENT for large-scale R-group sampling and generation
    3. Applying ADMET and target-activity models to filter generated molecules
    """
    
    def __init__(
        self,
        config: AgentConfig,
        mcp_client: MCPClientManager,
        additional_context: Optional[str] = None,
        medchem_retrieval_agent: Optional[Any] = None,
    ):
        """Initialize the creative molecule explorer agent.
        
        Args:
            config: Agent configuration
            mcp_client: MCP client manager
            additional_context: Optional user-provided context to inject at system prompt start
        """
        super().__init__(
            config,
            mcp_client,
            additional_context=additional_context,
            medchem_retrieval_agent=medchem_retrieval_agent,
        )
        self._tools: Optional[List[Any]] = None
        self._generation_outputs_by_round: Dict[str, Dict[str, Any]] = {}
        self._admet_outputs_by_round: Dict[str, Dict[str, Any]] = {}
        self._intersection_outputs_by_round: Dict[str, Dict[str, Any]] = {}
        self._intersection_generation_rounds: set[str] = set()
        self._intersection_generation_runs_by_round: Dict[str, int] = {}
        self._intersection_filter_attempts_by_round: Dict[str, int] = {}
        self._last_intersection_output_count_by_round: Dict[str, int] = {}
        self._nesso_reference_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
    
    @property
    def system_prompt(self) -> str:
        """Get the system prompt for this agent.
        
        Returns:
            Formatted system prompt with knowledge and property meta info
        """
        # Load property meta info
        property_meta_info = self._load_property_meta_info()
        
        return render_creative_system_prompt(
            property_meta_info=property_meta_info,
            additional_context=self.additional_context,
        )
    
    @property
    def output_schema(self) -> type | None:
        """Get the output schema for this agent.
        
        Returns:
            None because the creative agent now writes free-form reasoning.
        """
        return None

    @property
    def agent_role(self) -> str:
        return "creative"

    def _tool_description_for_wrapping(self, tool_name: str, description: str) -> str:
        if tool_name == "setup_generation_Mol2Mol_LinkInvent":
            routing_note = (
                "Creative routing guardrail: prefer `setup_libinvent` -> `prepare_scaffold` "
                "-> `run_generation` -> filtering for ordinary creative optimization. "
                "Use model_type=\"Mol2Mol\" only when the user explicitly asks for similar "
                "molecules, close analogs, or whole-molecule similarity exploration. "
                "Use model_type=\"LinkInvent\" only when the user explicitly asks for linker "
                "generation or linker replacement between specified fragments or warheads."
            )
        else:
            return super()._tool_description_for_wrapping(tool_name, description)
        if not description:
            return routing_note
        return f"{routing_note}\n\n{description}"
    
    async def get_tools(self) -> List[Any]:
        """Get tools for the creative molecule explorer.
        
        Returns:
            List of tools including R-group generation, ADMET filtering, and
            target-activity filtering.
        """
        if self._tools is None:
            all_tools = self.mcp_client.get_all_agent_tools()
            direct_tools = filter_tools_by_name(
                all_tools,
                CREATIVE_DIRECT_MCP_TOOL_NAMES,
            )
            self._tools = [*direct_tools]
            if self.medchem_retrieval_agent is not None:
                self._tools.append(self._build_ask_medchem_knowledge_tool())
            self._tools.append(self._build_similarity_mol2mol_tool())
            self._tools.append(self._build_setup_generation_tool())
            self._tools.append(self._build_setup_libinvent_tool())
            self._tools.append(self._build_run_generation_tool())
            self._tools.append(self._build_admet_filter_tool())
            self._tools.append(self._build_nesso_filter_tool())
            self._tools.append(self._build_constraint_filter_tool())
            self._tools.append(self._build_intersection_filter_tool())
        
        return self._tools

    def _build_similarity_mol2mol_tool(self) -> Any:
        @tool("generate_similarity_constrained_mol2mol")
        async def generate_similarity_constrained_mol2mol_tool(
            similarity_threshold: Optional[float] = None,
            device: str = "cuda:0",
        ) -> str:
            """Generate the current activity-optimization pool with Mol2Mol.

            M3OS binds the seed molecule and any explicit task similarity
            threshold. The backend requests up to 10000 unique qualifying
            candidates and registers the resulting CSV for exactly one final
            ADMET, Nesso, constraint, or intersection screening route.
            """
            if not self._target_activity_objective():
                raise ValueError(
                    "This Mol2Mol route is reserved for target-activity optimization."
                )
            round_key = self._pipeline_round_key()
            if self._intersection_screening_required():
                self._intersection_generation_rounds.add(round_key)
            if round_key in self._intersection_generation_rounds:
                self._ensure_intersection_generation_run_allowed(round_key)

            threshold = self._task_similarity_threshold(similarity_threshold)
            params = {
                "seed_smiles": self._activity_mol2mol_seed(),
                "similarity_threshold": threshold,
                "num_candidates": MOL2MOL_CANDIDATE_COUNT,
                "samples_per_round": MOL2MOL_SAMPLES_PER_ROUND,
                "max_rounds": MOL2MOL_MAX_ROUNDS,
                "device": str(device or "cuda:0").strip() or "cuda:0",
                "model_variant": "high_similarity",
                "temperature": 1.0,
                "random_seed": 42 + max(0, self._numeric_round_index() - 1),
                "exclude_seed": True,
            }
            result = await self._invoke_pipeline_tool(
                "generate_similarity_constrained_mol2mol",
                params,
                service_label="REINVENT MCP",
                recovery=(
                    "Use the authoritative activity-optimization seed and a valid "
                    "similarity threshold between zero and one."
                ),
            )
            payload = self._successful_csv_payload(
                result,
                path_field="output_csv_path",
                tool_name="generate_similarity_constrained_mol2mol",
                allowed_statuses={"success", "partial"},
            )
            if int(payload.get("output_count") or 0) < 1:
                raise RuntimeError(
                    "Mol2Mol produced no similarity-qualified candidates for screening."
                )
            payload["generated_csv_path"] = payload["output_csv_path"]
            self._generation_outputs_by_round[round_key] = payload
            if round_key in self._intersection_generation_rounds:
                self._intersection_generation_runs_by_round[round_key] = (
                    self._intersection_generation_runs_by_round.get(round_key, 0) + 1
                )
            self._admet_outputs_by_round.pop(round_key, None)
            self._intersection_outputs_by_round.pop(round_key, None)
            return self._result_text(result)

        return generate_similarity_constrained_mol2mol_tool

    def _build_setup_generation_tool(self) -> Any:
        @tool("setup_generation_Mol2Mol_LinkInvent")
        async def setup_generation_tool(
            model_type: Literal["Mol2Mol", "LinkInvent"],
            num_samples: int = 200,
            device: str = "cuda:0",
        ) -> str:
            """Set up Mol2Mol or LinkInvent generation for non-activity routes.

            Target-activity optimization uses
            generate_similarity_constrained_mol2mol instead. Mol2Mol here is
            for explicit whole-molecule similarity requests and
            LinkInvent is for explicit linker-generation requests.
            """
            if self._target_activity_objective():
                raise ValueError(
                    "Target-activity optimization must use "
                    "generate_similarity_constrained_mol2mol."
                )
            params = {
                "model_type": model_type,
                "num_samples": self._generation_sample_count(num_samples),
                "device": str(device or "cuda:0").strip() or "cuda:0",
            }
            result = await self._invoke_pipeline_tool(
                "setup_generation_Mol2Mol_LinkInvent",
                params,
                service_label="REINVENT MCP",
                recovery="Use Mol2Mol or LinkInvent with valid setup parameters.",
            )
            return self._result_text(result)

        return setup_generation_tool

    def _build_setup_libinvent_tool(self) -> Any:
        @tool("setup_libinvent")
        async def setup_libinvent_tool(
            num_samples: int = 200,
            device: str = "cuda:0",
        ) -> str:
            """Set up LibInvent generation for non-activity localized edits."""
            if self._target_activity_objective():
                raise ValueError(
                    "Target-activity optimization must use "
                    "generate_similarity_constrained_mol2mol."
                )
            params = {
                "num_samples": self._generation_sample_count(num_samples),
                "device": str(device or "cuda:0").strip() or "cuda:0",
            }
            result = await self._invoke_pipeline_tool(
                "setup_libinvent",
                params,
                service_label="REINVENT MCP",
                recovery="Re-call setup_libinvent with valid setup parameters.",
            )
            return self._result_text(result)

        return setup_libinvent_tool

    def _build_run_generation_tool(self) -> Any:
        @tool("run_generation")
        async def run_generation_tool(config_file: str) -> str:
            """Run REINVENT and register its real generated CSV for this MCGS round.

            Pass the exact configuration path returned by a REINVENT setup tool.
            Downstream filters automatically consume the successful output from
            this call, so do not construct or guess a candidate CSV path.
            """
            round_key = self._pipeline_round_key()
            if self._intersection_screening_required():
                self._intersection_generation_rounds.add(round_key)
            if round_key in self._intersection_generation_rounds:
                self._ensure_intersection_generation_run_allowed(round_key)
            result = await self._invoke_pipeline_tool(
                "run_generation",
                {"config_file": str(config_file or "").strip()},
                service_label="REINVENT MCP",
                recovery="Run the appropriate REINVENT setup tool and pass its exact config_file.",
            )
            payload = self._successful_csv_payload(
                result,
                path_field="generated_csv_path",
                tool_name="run_generation",
            )
            self._generation_outputs_by_round[round_key] = payload
            if round_key in self._intersection_generation_rounds:
                self._intersection_generation_runs_by_round[round_key] = (
                    self._intersection_generation_runs_by_round.get(round_key, 0) + 1
                )
            self._admet_outputs_by_round.pop(round_key, None)
            self._intersection_outputs_by_round.pop(round_key, None)
            return self._result_text(result)

        return run_generation_tool

    def _build_admet_filter_tool(self) -> Any:
        @tool("admet_filter_by_admetai")
        async def admet_filter_by_admetai_tool(
            preference_json: str,
            rand_str: str = "",
            candidate_csv_path: Optional[str] = None,
            top_n: int = 5,
        ) -> str:
            """Filter a candidate CSV using ADMET-AI preferences.

            Use this as the final route when the task has monotonic ADMET
            objectives and no fully specified absolute, interval, hold, or
            baseline-relative constraint. A property with only an
            increase/decrease direction is a ranking preference, not a runtime
            constraint; do not construct a task contract merely to use a
            constraint-screening tool.

            preference_json must be a JSON object string whose values are exactly
            "higher" or "lower", for example {"BBB_Martins": "higher"}.
            Do not use higher_better, lower_better, increase, decrease, improve,
            reduce, true, or false in tool arguments.
            Pass candidate_csv_path from run_generation or a previous filtering
            stage. Set top_n to the required stage output count.
            """
            rand_str = str(rand_str or "").strip()
            round_key = self._pipeline_round_key()
            generation_payload = self._generation_outputs_by_round.get(round_key)
            if generation_payload:
                resolved_csv_path = generation_payload["generated_csv_path"]
                rand_str = str(generation_payload.get("rand_str") or rand_str).strip()
            else:
                resolved_csv_path = self._resolve_candidate_csv_path_for_admet(
                    candidate_csv_path=candidate_csv_path,
                )
            if not rand_str and not resolved_csv_path:
                raise ValueError(
                    "Run REINVENT generation first or provide an existing candidate_csv_path "
                    "for ADMET filtering."
                )
            if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
                raise ValueError("top_n must be a positive integer.")
            top_n = DEFAULT_FINALIST_COUNT
            params: Dict[str, Any] = {
                "preference_json": self._normalize_admet_preference_json(preference_json),
                "rand_str": rand_str,
                "candidate_csv_path": resolved_csv_path or "",
                "top_n": top_n,
            }
            result = await self._invoke_admet_tool("admet_filter_by_admetai", params)
            payload = self._successful_csv_payload(
                result,
                path_field="output_csv_path",
                tool_name="admet_filter_by_admetai",
            )
            self._admet_outputs_by_round[round_key] = payload
            return self._result_text(result)

        return admet_filter_by_admetai_tool

    def _build_nesso_filter_tool(self) -> Any:
        @tool("nesso_filter_by_cofolding")
        async def nesso_filter_by_cofolding_tool(
            protein_sequence: str,
            rand_str: str = "",
            protein_id: str = "target",
            candidate_csv_path: str = "",
            top_n: int = 5,
        ) -> str:
            """Return the ten current-round candidates ranked by Nesso probability.

            M3OS binds the successful current-round generation CSV and the
            authoritative task protein input. This route is for activity-only
            tasks without an explicit lead-optimization task contract.
            """
            del protein_sequence, rand_str, candidate_csv_path, top_n
            round_key = self._pipeline_round_key()
            generation_payload = self._generation_outputs_by_round.get(round_key)
            if not generation_payload:
                raise ValueError(
                    "No successful current-round candidate CSV is registered. "
                    "Run REINVENT generation first."
                )
            params = {
                "protein_id": str(protein_id or "target").strip() or "target",
                "candidate_csv_path": generation_payload["generated_csv_path"],
                "top_n": ACTIVITY_FINALIST_COUNT,
                **self._protein_input_params(),
            }
            result = await self._invoke_pipeline_tool(
                "nesso_filter_by_cofolding",
                params,
                service_label="Nesso MCP",
                recovery="Use the current-round REINVENT output and authoritative target protein input.",
            )
            self._successful_csv_payload(
                result,
                path_field="output_csv_path",
                tool_name="nesso_filter_by_cofolding",
            )
            return self._result_text(result)

        return nesso_filter_by_cofolding_tool

    def _build_constraint_filter_tool(self) -> Any:
        @tool("screen_reinvent_candidates_by_constraints")
        async def screen_reinvent_candidates_by_constraints_tool(
            task_contract_json: str = "",
            top_n: int = DEFAULT_FINALIST_COUNT,
        ) -> str:
            """Select five candidates satisfying runtime hard, hold, and ADMET constraints.

            This is not a general ADMET-ranking route. Use it only when the task
            has at least one fully specified, directly evaluable constraint and
            no target-activity requirement: an explicit operator and value or
            interval, a hold with baseline and tolerance, or a baseline-relative
            objective with an exact source/endpoint, baseline, direction, and
            numeric threshold. A bare property plus increase/decrease direction
            is not a constraint, even if it appears in an attached task contract;
            use admet_filter_by_admetai for such monotonic ADMET objectives.

            Supply task_contract_json only when no structured contract was
            attached to the session. Include only complete constraint fields
            stated by the current task; never invent an operator, endpoint,
            baseline, tolerance, or threshold to make this tool applicable.
            """
            del top_n
            round_key = self._pipeline_round_key()
            generation_payload = self._generation_outputs_by_round.get(round_key)
            if not generation_payload:
                raise ValueError("Run REINVENT generation before constraint screening.")
            contract = self._resolve_task_contract(task_contract_json)
            reference_smiles = self._contract_reference_smiles(contract)
            result = await self._invoke_pipeline_tool(
                "admet_filter_by_task_constraints",
                {
                    "task_contract_json": json.dumps(contract, ensure_ascii=False),
                    "reference_smiles": reference_smiles,
                    "candidate_csv_path": generation_payload["generated_csv_path"],
                },
                service_label="ADMET MCP",
                recovery="Use a valid runtime task contract and the current-round REINVENT CSV.",
            )
            payload = self._successful_csv_payload(
                result,
                path_field="output_csv_path",
                tool_name="admet_filter_by_task_constraints",
            )
            selected = self._limit_constraint_candidates(
                payload,
                generated_csv_path=generation_payload["generated_csv_path"],
            )
            self._intersection_outputs_by_round[round_key] = selected
            return json.dumps(selected, ensure_ascii=False)

        return screen_reinvent_candidates_by_constraints_tool

    def _build_intersection_filter_tool(self) -> Any:
        @tool("screen_reinvent_candidates_by_intersection")
        async def screen_reinvent_candidates_by_intersection_tool(
            task_contract_json: str = "",
            top_n: int = ACTIVITY_FINALIST_COUNT,
        ) -> str:
            """Select candidates from the hard/ADMET and activity-qualified sets.

            The backend evaluates the complete hard/ADMET-qualified set with
            Nesso probability and affinity gates, without truncating that set
            before screening. Each Creative round uses one 10000-sample pool and
            one intersection screen. An empty intersection remains the final
            screened result for that round; generation is not retried and no
            threshold is relaxed. The returned qualified-set context may guide
            unverified next-step analogs for Critic evaluation.
            """
            del top_n
            round_key = self._pipeline_round_key()
            generation_payload = self._generation_outputs_by_round.get(round_key)
            if not generation_payload:
                raise ValueError("Run REINVENT generation before intersection screening.")
            self._register_intersection_generation_round(round_key)
            self._ensure_intersection_filter_allowed(round_key)
            contract = self._resolve_task_contract(task_contract_json)
            reference_smiles = self._contract_reference_smiles(contract)
            generated_csv = generation_payload["generated_csv_path"]

            admet_result = await self._invoke_pipeline_tool(
                "admet_filter_by_task_constraints",
                {
                    "task_contract_json": json.dumps(contract, ensure_ascii=False),
                    "reference_smiles": reference_smiles,
                    "candidate_csv_path": generated_csv,
                },
                service_label="ADMET MCP",
                recovery="Use the runtime task contract and current-round REINVENT CSV.",
            )
            admet_payload = self._successful_csv_payload(
                admet_result,
                path_field="output_csv_path",
                tool_name="admet_filter_by_task_constraints",
            )

            admet_pass_count = int(admet_payload.get("output_count") or 0)
            if admet_pass_count > 0:
                reference_nesso = await self._predict_nesso_reference(reference_smiles)
                activity_payload = await self._rank_csv_with_nesso(
                    admet_payload["output_csv_path"],
                    protein_id="target",
                )
                payload = self._select_nesso_activity_candidates(
                    activity_payload,
                    reference_nesso=reference_nesso,
                    contract=contract,
                    generated_csv_path=generated_csv,
                    admet_payload=admet_payload,
                )
            else:
                payload = self._empty_nesso_intersection_payload(
                    generated_csv_path=generated_csv,
                    admet_payload=admet_payload,
                )
            attempt_number = self._intersection_filter_attempts_by_round.get(round_key, 0) + 1
            intersection_count = int(payload.get("intersection_count") or 0)
            output_count = int(payload.get("output_count") or 0)
            if intersection_count > 0:
                screening_outcome = "intersection_candidates_found"
            else:
                screening_outcome = "empty_intersection"
            payload.update(
                {
                    "generation_attempt": attempt_number,
                    "maximum_generation_attempts": INTERSECTION_MAX_ATTEMPTS,
                    "retry_allowed": False,
                    "next_generation_sample_count": None,
                    "screening_outcome": screening_outcome,
                    "fallback_applied": False,
                }
            )
            payload["screening_context"]["mcgs_round"] = self._numeric_round_index()
            self._intersection_filter_attempts_by_round[round_key] = attempt_number
            self._last_intersection_output_count_by_round[round_key] = output_count
            self._intersection_outputs_by_round[round_key] = payload
            return json.dumps(payload, ensure_ascii=False)

        return screen_reinvent_candidates_by_intersection_tool

    def screening_context_for_round(self, round_value: Any) -> Dict[str, Any]:
        """Return the completed screening context for one MCGS round."""
        payload = self._intersection_outputs_by_round.get(str(round_value)) or {}
        context = payload.get("screening_context")
        if not isinstance(context, Mapping):
            return {}
        return json.loads(json.dumps(dict(context), ensure_ascii=False, default=str))

    def _resolve_task_contract(self, task_contract_json: str) -> Dict[str, Any]:
        attached = self.event_context.get("leadopt_task_contract")
        if isinstance(attached, dict) and attached:
            return dict(attached)
        try:
            parsed = json.loads(str(task_contract_json or ""))
        except json.JSONDecodeError as exc:
            raise ValueError("task_contract_json must be a JSON object string.") from exc
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError(
                "Constraint screening requires a task contract attached to the session "
                "or supplied as task_contract_json."
            )
        return parsed

    def _contract_reference_smiles(self, contract: Mapping[str, Any]) -> str:
        reference_smiles = str(
            contract.get("reference_smiles")
            or self.event_context.get("initial_smiles")
            or ""
        ).strip()
        if not reference_smiles:
            raise ValueError("Constraint screening requires the initial molecule SMILES.")
        return reference_smiles

    def _activity_mol2mol_seed(self) -> str:
        contract = self.event_context.get("leadopt_task_contract")
        if (
            isinstance(contract, Mapping)
            and self._contract_has_similarity_constraint(contract)
        ):
            seed_smiles = str(
                contract.get("reference_smiles")
                or self.event_context.get("initial_smiles")
                or ""
            ).strip()
        else:
            seed_smiles = str(
                self.event_context.get("current_smiles")
                or self.event_context.get("initial_smiles")
                or ""
            ).strip()
        if not seed_smiles:
            raise ValueError("Mol2Mol activity generation requires a seed molecule SMILES.")
        return seed_smiles

    @staticmethod
    def _similarity_constraint_entries(
        contract: Mapping[str, Any],
    ) -> List[Mapping[str, Any]]:
        similarity_properties = {
            "similarity",
            "molecular_similarity",
            "tanimoto_similarity",
            "smdd_tanimoto",
        }
        entries: List[Mapping[str, Any]] = []
        for section_name in ("hard_constraints", "hold_constant", "holds"):
            for entry in contract.get(section_name, []) or []:
                if not isinstance(entry, Mapping):
                    continue
                property_name = (
                    str(entry.get("property") or "").casefold().replace("-", "_")
                )
                if property_name in similarity_properties:
                    entries.append(entry)
        return entries

    def _contract_has_similarity_constraint(
        self, contract: Mapping[str, Any]
    ) -> bool:
        return bool(self._similarity_constraint_entries(contract))

    def _task_similarity_threshold(self, requested: Optional[float]) -> float:
        contract = self.event_context.get("leadopt_task_contract")
        if isinstance(contract, Mapping):
            for entry in self._similarity_constraint_entries(contract):
                operator = str(
                    entry.get("operator") or entry.get("comparator") or ""
                ).casefold()
                if operator not in {"gt", "gte", ">", ">="}:
                    continue
                raw_value = entry.get("value", entry.get("threshold"))
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if not 0.0 <= value <= 1.0:
                    raise ValueError("Task similarity threshold must be between zero and one.")
                return value
        value = (
            MOL2MOL_DEFAULT_SIMILARITY_THRESHOLD
            if requested is None
            else float(requested)
        )
        if not 0.0 <= value <= 1.0:
            raise ValueError("similarity_threshold must be between zero and one.")
        return value

    def _numeric_round_index(self) -> int:
        try:
            return int(self.event_context.get("mcgs_round") or 1)
        except (TypeError, ValueError):
            return 1

    def _protein_input_params(self) -> Dict[str, str]:
        fasta_value = str(self.event_context.get("protein_fasta_path") or "").strip()
        if fasta_value:
            fasta = Path(fasta_value).expanduser().resolve()
            if not fasta.is_file():
                raise ValueError(f"Task protein FASTA does not exist: {fasta}")
            return {"protein_fasta_path": str(fasta)}
        sequence = self._task_protein_sequence()
        if not sequence:
            raise ValueError("Target-activity screening requires a target protein input.")
        return {"protein_sequence": sequence}

    @staticmethod
    def _activity_probability_threshold(contract: Dict[str, Any]) -> float:
        for entry in contract.get("hard_constraints") or []:
            if not isinstance(entry, dict):
                continue
            property_name = str(entry.get("property") or "").casefold()
            source = str(entry.get("source") or "").casefold().replace("-", "_")
            probability_properties = {
                "binding_probability",
                "interaction_probability",
                "activity_probability",
                "boltz_binding_probability",
                "transformercpi2_interaction_probability",
                "nesso_affinity_probability_binary",
            }
            if source not in {"activity", "target_activity", "binding"} and property_name not in probability_properties:
                continue
            operator = str(entry.get("operator") or entry.get("comparator") or "").casefold()
            if operator in {"lt", "lte", "<", "<="}:
                raise ValueError(
                    "Nesso intersection screening supports minimum activity "
                    "probability constraints, not maximum-probability constraints."
                )
            value = entry.get("value")
            if value is not None:
                numeric = float(value)
                if 0.0 <= numeric <= 1.0:
                    return numeric
                raise ValueError("Activity probability threshold must be between 0 and 1.")
            match = re.search(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", str(entry.get("threshold") or ""))
            if match:
                return float(match.group(0))
        return 0.0

    def _limit_constraint_candidates(
        self,
        payload: Dict[str, Any],
        *,
        generated_csv_path: str,
    ) -> Dict[str, Any]:
        eligible = pd.read_csv(payload["output_csv_path"])
        if "SMILES" not in eligible.columns:
            raise RuntimeError("Constraint filtering output is missing the SMILES column.")
        selected = eligible.head(DEFAULT_FINALIST_COUNT).copy()
        output_dir = PROJECT_TMP_DIR / "leadopt_constraints"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = (output_dir / f"leadopt_constraints_{uuid.uuid4().hex}.csv").resolve()
        selected.to_csv(output_csv, index=False)
        candidates = []
        for _, row in selected.iterrows():
            candidate = {"SMILES": str(row["SMILES"])}
            for field in ("Scaffold", "Input_SMILES", "Warheads", "R-groups", "Linker"):
                value = self._csv_text(row.get(field))
                if value:
                    candidate[field] = value
            candidates.append(candidate)
        return {
            "status": "success",
            "stage": "leadopt_constraints",
            "input_csv_path": str(Path(generated_csv_path).resolve()),
            "eligible_csv_path": payload["output_csv_path"],
            "output_csv_path": str(output_csv),
            "input_count": int(payload.get("input_count") or 0),
            "eligible_count": int(payload.get("output_count") or 0),
            "output_count": int(len(selected)),
            "candidates": candidates,
        }

    @staticmethod
    def _canonical_csv_smiles(value: Any) -> str:
        valid, canonical, _error = validate_smiles(str(value or "").strip())
        return canonical if valid and canonical else ""

    @staticmethod
    def _csv_text(value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value)

    async def _rank_csv_with_nesso(
        self,
        candidate_csv_path: str,
        *,
        protein_id: str,
    ) -> Dict[str, Any]:
        frame = pd.read_csv(candidate_csv_path, usecols=["SMILES"])
        candidate_count = int(len(frame))
        if candidate_count < 1:
            raise ValueError("Nesso activity screening requires at least one candidate.")
        result = await self._invoke_pipeline_tool(
            "nesso_filter_by_cofolding",
            {
                "protein_id": str(protein_id or "target").strip() or "target",
                "candidate_csv_path": str(Path(candidate_csv_path).resolve()),
                "top_n": candidate_count,
                **self._protein_input_params(),
            },
            service_label="Nesso MCP",
            recovery="Use the candidate CSV and authoritative target protein input.",
        )
        return self._successful_csv_payload(
            result,
            path_field="output_csv_path",
            tool_name="nesso_filter_by_cofolding",
        )

    async def _predict_nesso_reference(self, reference_smiles: str) -> Dict[str, Any]:
        protein_params = self._protein_input_params()
        protein_identity = str(
            protein_params.get("protein_fasta_path")
            or protein_params.get("protein_sequence")
            or ""
        )
        reference_key = self._canonical_csv_smiles(reference_smiles)
        cache_key = (protein_identity, reference_key)
        cached = self._nesso_reference_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        result = await self._invoke_pipeline_tool(
            "nesso_predict_by_cofolding",
            {
                "smiles_list": json.dumps([reference_smiles], ensure_ascii=False),
                "protein_id": "target",
                **protein_params,
            },
            service_label="Nesso MCP",
            recovery="Use the initial molecule and authoritative target protein input.",
        )
        records = self._parse_nesso_prediction_records(result)
        if len(records) != 1:
            raise RuntimeError("Nesso returned no unique prediction for the initial molecule.")
        self._nesso_reference_cache[cache_key] = dict(records[0])
        return dict(records[0])

    @staticmethod
    def _parse_nesso_prediction_records(result: Any) -> List[Dict[str, Any]]:
        if isinstance(result, list):
            parsed = result
        else:
            parsed: Any = None
            text = str(result or "").strip()
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    break
                except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                    continue
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            raise RuntimeError("Nesso returned an invalid prediction record list.")
        return [dict(item) for item in parsed]

    @staticmethod
    def _nesso_affinity_task_margin(
        contract: Mapping[str, Any],
        candidate_affinity: float,
        reference_affinity: float,
    ) -> float | None:
        objectives = contract.get("optimization_objectives", contract.get("objectives", []))
        holds = contract.get("hold_constant", contract.get("holds", []))
        for entry in objectives or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("property") or "").casefold() != "binding_affinity":
                continue
            threshold = float(entry.get("threshold", 0.0))
            direction = str(entry.get("direction") or "minimize").casefold()
            improvement = (
                candidate_affinity - reference_affinity
                if direction == "maximize"
                else reference_affinity - candidate_affinity
            )
            return improvement - threshold
        for entry in holds or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("property") or "").casefold() != "binding_affinity":
                continue
            tolerance = float(entry.get("tolerance", 0.0))
            direction = str(entry.get("direction") or "change").casefold()
            delta = candidate_affinity - reference_affinity
            if direction == "increase":
                return tolerance - delta
            if direction == "decrease":
                return tolerance + delta
            return tolerance - abs(delta)
        return None

    @staticmethod
    def _nesso_affinity_task_gate(
        contract: Mapping[str, Any],
        candidate_affinity: float,
        reference_affinity: float,
    ) -> bool:
        margin = CreativeMoleculeExplorerAgent._nesso_affinity_task_margin(
            contract,
            candidate_affinity,
            reference_affinity,
        )
        return margin is None or margin + NESSO_AFFINITY_GATE_EPSILON >= 0.0

    @staticmethod
    def _screening_scalar(value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        return value if isinstance(value, (str, int, float, bool)) else str(value)

    @classmethod
    def _screening_candidate_record(cls, row: pd.Series) -> Dict[str, Any]:
        record = {
            str(column): cls._screening_scalar(value)
            for column, value in row.to_dict().items()
            if str(column) not in SCREENING_CONTEXT_EXCLUDED_COLUMNS
            and str(column) != "_canonical_smiles"
        }
        for field in (
            "hard_constraint_details",
            "admet_constraint_details",
            "deferred_activity_constraints",
        ):
            value = record.get(field)
            if not isinstance(value, str):
                continue
            try:
                record[field] = json.loads(value)
            except json.JSONDecodeError:
                pass
        probability = float(record[NESSO_PROBABILITY_COLUMN])
        probability_margin = float(record["Nesso_probability_target_margin"])
        affinity_margin_value = record.get("Nesso_affinity_target_margin")
        affinity_margin = (
            None if affinity_margin_value is None else float(affinity_margin_value)
        )
        gaps = {
            NESSO_PROBABILITY_COLUMN: float(max(0.0, -probability_margin)),
            NESSO_AFFINITY_COLUMN: (
                0.0 if affinity_margin is None else float(max(0.0, -affinity_margin))
            ),
        }
        failures: List[Dict[str, Any]] = []
        if not bool(record.get("Nesso_probability_task_pass")):
            failures.append(
                {
                    "property": NESSO_PROBABILITY_COLUMN,
                    "actual": probability,
                    "operator": "gt",
                    "target": float(record["Nesso_probability_target"]),
                    "margin": probability_margin,
                    "gap": gaps[NESSO_PROBABILITY_COLUMN],
                }
            )
        if not bool(record.get("Nesso_affinity_task_pass")):
            failures.append(
                {
                    "property": NESSO_AFFINITY_COLUMN,
                    "actual": float(record[NESSO_AFFINITY_COLUMN]),
                    "reference": float(record["Nesso_reference_affinity_pred_value"]),
                    "margin": affinity_margin,
                    "gap": gaps[NESSO_AFFINITY_COLUMN],
                }
            )
        record["target_gaps"] = gaps
        record["failure_reasons"] = failures
        return record

    def _select_nesso_activity_candidates(
        self,
        activity_payload: Dict[str, Any],
        *,
        reference_nesso: Mapping[str, Any],
        contract: Mapping[str, Any],
        generated_csv_path: str,
        admet_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        activity_frame = pd.read_csv(activity_payload["output_csv_path"])
        required_columns = {"SMILES", NESSO_PROBABILITY_COLUMN, NESSO_AFFINITY_COLUMN}
        missing_columns = sorted(required_columns.difference(activity_frame.columns))
        if missing_columns:
            raise RuntimeError(
                "Nesso filtering output is missing required columns: "
                + ", ".join(missing_columns)
            )
        activity_frame["_canonical_smiles"] = activity_frame["SMILES"].map(
            self._canonical_csv_smiles
        )
        for column in (NESSO_PROBABILITY_COLUMN, NESSO_AFFINITY_COLUMN):
            activity_frame[column] = pd.to_numeric(activity_frame[column], errors="coerce")
        activity_frame = activity_frame.loc[
            (activity_frame["_canonical_smiles"] != "")
            & activity_frame[NESSO_PROBABILITY_COLUMN].notna()
            & activity_frame[NESSO_AFFINITY_COLUMN].notna()
        ].drop_duplicates("_canonical_smiles", keep="first")

        reference_probability = float(reference_nesso[NESSO_PROBABILITY_COLUMN])
        reference_affinity = float(reference_nesso[NESSO_AFFINITY_COLUMN])
        probability_threshold = self._activity_probability_threshold(dict(contract))
        activity_frame["Nesso_affinity_improvement_vs_initial"] = (
            reference_affinity - activity_frame[NESSO_AFFINITY_COLUMN]
        )
        activity_frame["Nesso_probability_target"] = probability_threshold
        activity_frame["Nesso_probability_target_margin"] = (
            activity_frame[NESSO_PROBABILITY_COLUMN] - probability_threshold
        )
        activity_frame["Nesso_probability_task_pass"] = (
            activity_frame[NESSO_PROBABILITY_COLUMN] > probability_threshold
        )
        activity_frame["Nesso_reference_affinity_pred_value"] = reference_affinity
        activity_frame["Nesso_affinity_target_margin"] = activity_frame[
            NESSO_AFFINITY_COLUMN
        ].map(
            lambda value: self._nesso_affinity_task_margin(
                contract,
                float(value),
                reference_affinity,
            )
        )
        activity_frame["Nesso_affinity_task_pass"] = activity_frame[
            NESSO_AFFINITY_COLUMN
        ].map(
            lambda value: self._nesso_affinity_task_gate(
                contract,
                float(value),
                reference_affinity,
            )
        )
        activity_frame["Nesso_activity_constraints_pass"] = (
            activity_frame["Nesso_probability_task_pass"]
            & activity_frame["Nesso_affinity_task_pass"]
        )
        activity_frame["Nesso_activity_failed_constraint_count"] = (
            (~activity_frame["Nesso_probability_task_pass"]).astype(int)
            + (~activity_frame["Nesso_affinity_task_pass"]).astype(int)
        )
        activity_frame["Nesso_probability_target_gap"] = (
            -activity_frame["Nesso_probability_target_margin"]
        ).clip(lower=0.0)
        activity_frame["Nesso_affinity_target_gap"] = pd.to_numeric(
            activity_frame["Nesso_affinity_target_margin"], errors="coerce"
        ).map(lambda value: 0.0 if pd.isna(value) else max(0.0, -float(value)))
        strict_eligible = activity_frame.loc[
            activity_frame["Nesso_activity_constraints_pass"] == True  # noqa: E712
        ].sort_values(
            ["Nesso_affinity_improvement_vs_initial", NESSO_PROBABILITY_COLUMN],
            ascending=[False, False],
            kind="stable",
        )
        eligible = strict_eligible
        selected = eligible.head(ACTIVITY_FINALIST_COUNT).drop(
            columns=["_canonical_smiles"], errors="ignore"
        )
        output_dir = PROJECT_TMP_DIR / "leadopt_intersections"
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex
        activity_pass_csv = (output_dir / f"nesso_activity_pass_{suffix}.csv").resolve()
        output_name = "leadopt_intersection"
        output_csv = (output_dir / f"{output_name}_{suffix}.csv").resolve()
        eligible.drop(columns=["_canonical_smiles"], errors="ignore").to_csv(
            activity_pass_csv,
            index=False,
        )
        selected.to_csv(output_csv, index=False)
        context_frame = activity_frame.drop(columns=["_canonical_smiles"], errors="ignore")
        admet_hard_pass_candidates = [
            self._screening_candidate_record(row)
            for _, row in context_frame.iterrows()
        ]
        activity_pass_candidates = [
            self._screening_candidate_record(row)
            for _, row in strict_eligible.drop(
                columns=["_canonical_smiles"], errors="ignore"
            ).iterrows()
        ]
        closest_frame = activity_frame.loc[
            activity_frame["Nesso_activity_constraints_pass"] == False  # noqa: E712
        ].sort_values(
            [
                "Nesso_activity_failed_constraint_count",
                "Nesso_affinity_target_gap",
                "Nesso_probability_target_gap",
                "Nesso_affinity_improvement_vs_initial",
                NESSO_PROBABILITY_COLUMN,
            ],
            ascending=[True, True, True, False, False],
            kind="stable",
        ).head(SCREENING_CONTEXT_NEAREST_COUNT)
        closest_to_activity = [
            self._screening_candidate_record(row)
            for _, row in closest_frame.drop(
                columns=["_canonical_smiles"], errors="ignore"
            ).iterrows()
        ]
        candidates = []
        for _, row in selected.iterrows():
            candidate = {
                "SMILES": str(row["SMILES"]),
                NESSO_PROBABILITY_COLUMN: float(row[NESSO_PROBABILITY_COLUMN]),
                NESSO_AFFINITY_COLUMN: float(row[NESSO_AFFINITY_COLUMN]),
                "Nesso_affinity_improvement_vs_initial": float(
                    row["Nesso_affinity_improvement_vs_initial"]
                ),
            }
            template = next(
                (
                    self._csv_text(row.get(column))
                    for column in ("Scaffold", "Input_SMILES", "Warheads")
                    if self._csv_text(row.get(column))
                ),
                "",
            )
            groups = next(
                (
                    self._csv_text(row.get(column))
                    for column in ("R-groups", "Linker")
                    if self._csv_text(row.get(column))
                ),
                "",
            )
            if template:
                candidate["Template"] = template
            if groups:
                candidate["R-groups"] = groups
            candidates.append(candidate)
        eligible_count = int(len(eligible))
        payload = {
            "status": "success",
            "stage": output_name,
            "input_csv_path": str(Path(generated_csv_path).resolve()),
            "admet_hard_pass_csv_path": admet_payload["output_csv_path"],
            "activity_scored_csv_path": activity_payload["output_csv_path"],
            "activity_pass_csv_path": str(activity_pass_csv),
            "output_csv_path": str(output_csv),
            "input_count": int(admet_payload.get("input_count") or 0),
            "admet_hard_pass_count": int(admet_payload.get("output_count") or 0),
            "activity_evaluated_count": int(activity_payload.get("input_count") or 0),
            "activity_pass_count": eligible_count,
            "strict_activity_pass_count": int(len(strict_eligible)),
            "intersection_count": eligible_count,
            "output_count": int(len(selected)),
            "reference_probability": reference_probability,
            "reference_affinity_pred_value": reference_affinity,
            "probability_proxy_threshold": probability_threshold,
            "probability_requires_better_than_initial": False,
            "ranking_metric": "Nesso_affinity_improvement_vs_initial",
            "combined_admet_and_hard_constraints_satisfied": True,
            "candidates": candidates,
            "screening_context": {
                "admet_hard_pass_candidates": admet_hard_pass_candidates,
                "activity_pass_candidates": activity_pass_candidates,
                "closest_to_activity_candidates": closest_to_activity,
                "closest_to_admet_hard_candidates": [],
                "intersection_candidates": activity_pass_candidates,
                "intersection_count": eligible_count,
            },
        }
        return payload

    def _empty_nesso_intersection_payload(
        self,
        *,
        generated_csv_path: str,
        admet_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        output_dir = PROJECT_TMP_DIR / "leadopt_intersections"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = (output_dir / f"leadopt_intersection_{uuid.uuid4().hex}.csv").resolve()
        pd.DataFrame(columns=["SMILES", NESSO_PROBABILITY_COLUMN, NESSO_AFFINITY_COLUMN]).to_csv(
            output_csv,
            index=False,
        )
        return {
            "status": "success",
            "stage": "leadopt_intersection",
            "input_csv_path": str(Path(generated_csv_path).resolve()),
            "admet_hard_pass_csv_path": admet_payload["output_csv_path"],
            "activity_pass_csv_path": str(output_csv),
            "output_csv_path": str(output_csv),
            "input_count": int(admet_payload.get("input_count") or 0),
            "admet_hard_pass_count": 0,
            "activity_evaluated_count": 0,
            "activity_pass_count": 0,
            "intersection_count": 0,
            "output_count": 0,
            "ranking_metric": NESSO_PROBABILITY_COLUMN,
            "combined_admet_and_hard_constraints_satisfied": True,
            "candidates": [],
            "screening_context": {
                "admet_hard_pass_candidates": [],
                "activity_pass_candidates": [],
                "closest_to_activity_candidates": [],
                "closest_to_admet_hard_candidates": [],
                "intersection_candidates": [],
                "intersection_count": 0,
            },
        }

    async def _invoke_admet_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        return await self._invoke_pipeline_tool(
            tool_name,
            params,
            service_label="ADMET MCP",
            recovery=(
                "Re-call admet_filter_by_admetai with the candidate CSV from the "
                "preceding pipeline stage and preference_json values that are exactly "
                "\"higher\" or \"lower\"."
            ),
        )

    async def _invoke_pipeline_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        *,
        service_label: str,
        recovery: str,
    ) -> Any:
        try:
            return await self.mcp_client.invoke_tool(tool_name, params)
        except Exception as exc:
            raise RuntimeError(
                self._format_mcp_failure(
                    tool_name,
                    exc,
                    service_label=service_label,
                    recovery=recovery,
                )
            ) from exc

    def _pipeline_round_key(self) -> str:
        value = self.event_context.get("mcgs_round")
        if value is None:
            value = self.event_context.get("round", "unknown")
        return str(value)

    def _task_protein_sequence(self) -> str:
        raw_sequence = str(self.event_context.get("initial_protein_sequence") or "")
        residues = []
        for line in raw_sequence.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(">"):
                residues.append("".join(stripped.split()))
        return "".join(residues).upper()

    def _generation_sample_count(self, requested_count: int) -> int:
        if not isinstance(requested_count, int) or isinstance(requested_count, bool):
            raise ValueError("num_samples must be a positive integer.")
        if requested_count < 1:
            raise ValueError("num_samples must be a positive integer.")
        round_key = self._pipeline_round_key()
        explicit_intersection_count = (
            requested_count in INTERSECTION_GENERATION_SAMPLE_COUNTS
            and self._target_activity_objective()
        )
        if (
            self._intersection_screening_required()
            or round_key in self._intersection_generation_rounds
            or explicit_intersection_count
        ):
            self._intersection_generation_rounds.add(round_key)
            return self._next_intersection_generation_sample_count(round_key)
        if self._target_activity_objective():
            return ACTIVITY_GENERATION_SAMPLE_COUNT
        return requested_count

    def _register_intersection_generation_round(self, round_key: str) -> None:
        self._intersection_generation_rounds.add(round_key)
        if round_key not in self._intersection_generation_runs_by_round:
            attempts = self._intersection_filter_attempts_by_round.get(round_key, 0)
            self._intersection_generation_runs_by_round[round_key] = attempts + 1

    def _next_intersection_generation_sample_count(self, round_key: str) -> int:
        attempts = self._intersection_filter_attempts_by_round.get(round_key, 0)
        runs = self._intersection_generation_runs_by_round.get(round_key, 0)
        if attempts > 0 and self._last_intersection_output_count_by_round.get(round_key, 0) > 0:
            raise ValueError(
                "Intersection screening already found candidates for this MCGS round. "
                "Return them without another generation attempt."
            )
        if attempts >= INTERSECTION_MAX_ATTEMPTS:
            raise ValueError(
                "Intersection screening permits a single generation attempt per MCGS "
                "round. Return the existing result without further generation."
            )
        if runs > attempts:
            raise ValueError(
                "The current intersection-generation pool has not been screened yet. "
                "Run screen_reinvent_candidates_by_intersection before another setup."
            )
        return INTERSECTION_GENERATION_SAMPLE_COUNTS[attempts]

    def _ensure_intersection_generation_run_allowed(self, round_key: str) -> None:
        self._next_intersection_generation_sample_count(round_key)

    def _ensure_intersection_filter_allowed(self, round_key: str) -> None:
        attempts = self._intersection_filter_attempts_by_round.get(round_key, 0)
        runs = self._intersection_generation_runs_by_round.get(round_key, 0)
        if attempts >= INTERSECTION_MAX_ATTEMPTS:
            raise ValueError(
                "Intersection screening permits a single generation attempt per MCGS round."
            )
        if runs != attempts + 1:
            if runs <= attempts:
                raise ValueError("Run a fresh REINVENT generation before intersection screening.")
            raise ValueError(
                "Only one unscreened REINVENT pool is allowed per intersection attempt."
            )

    def _intersection_screening_required(self) -> bool:
        contract = self.event_context.get("leadopt_task_contract")
        if not isinstance(contract, dict) or not contract:
            return self._combined_admet_activity_objective()

        activity_properties = {
            "binding_affinity",
            "binding_probability",
            "interaction_probability",
            "activity_probability",
            "boltz_binding_probability",
            "transformercpi2_interaction_probability",
            "nesso_affinity_probability_binary",
            "nesso_affinity_pred_value",
        }

        def is_activity(entry: Any) -> bool:
            if not isinstance(entry, dict):
                return False
            source = str(entry.get("source") or "").casefold().replace("-", "_")
            property_name = str(entry.get("property") or "").casefold()
            return source in {"activity", "target_activity", "binding"} or property_name in activity_properties

        entries = [
            *(contract.get("optimization_objectives", contract.get("objectives", [])) or []),
            *(contract.get("hold_constant", contract.get("holds", [])) or []),
            *(contract.get("hard_constraints", []) or []),
        ]
        return any(is_activity(entry) for entry in entries) and any(
            isinstance(entry, dict) and not is_activity(entry)
            for entry in entries
        )

    def _target_activity_objective(self) -> bool:
        contract = self.event_context.get("leadopt_task_contract")
        contract_has_activity = False
        if isinstance(contract, dict):
            activity_properties = {
                "binding_affinity",
                "binding_probability",
                "interaction_probability",
                "activity_probability",
                "boltz_binding_probability",
                "transformercpi2_interaction_probability",
                "nesso_affinity_probability_binary",
                "nesso_affinity_pred_value",
            }
            for section in (
                contract.get("optimization_objectives", contract.get("objectives", [])),
                contract.get("hold_constant", contract.get("holds", [])),
                contract.get("hard_constraints", []),
            ):
                contract_has_activity = contract_has_activity or any(
                    isinstance(entry, dict)
                    and (
                        str(entry.get("source") or "").casefold().replace("-", "_")
                        in {"activity", "target_activity", "binding"}
                        or str(entry.get("property") or "").casefold()
                        in activity_properties
                    )
                    for entry in (section or [])
                )
        task_text = " ".join(
            str(self.event_context.get(key) or "")
            for key in ("optimization_goal", "project_manager_brief")
        ).casefold()
        activity_terms = (
            "affinity",
            "binding",
            "activity",
            "potency",
            "ic50",
            "ec50",
            "ki ",
            "kd ",
            "nesso",
        )
        has_protein = bool(
            self._task_protein_sequence()
            or str(self.event_context.get("protein_fasta_path") or "").strip()
        )
        return has_protein and (
            contract_has_activity or any(term in task_text for term in activity_terms)
        )

    def _combined_admet_activity_objective(self) -> bool:
        properties = self.event_context.get("selected_admet_properties")
        has_admet = isinstance(properties, (list, tuple, set)) and any(
            str(item).strip() for item in properties
        )
        preference_json = str(
            self.event_context.get("selected_admet_preference_json") or ""
        ).strip()
        has_admet = has_admet or preference_json not in {"", "{}"}
        return has_admet and self._target_activity_objective()

    @staticmethod
    def _result_text(result: Any) -> str:
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    def _successful_csv_payload(
        self,
        result: Any,
        *,
        path_field: str,
        tool_name: str,
        allowed_statuses: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        if isinstance(result, dict):
            payload = dict(result)
        else:
            try:
                payload = json.loads(str(result))
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"{tool_name} returned invalid JSON output.") from exc
        statuses = allowed_statuses or {"success"}
        if not isinstance(payload, dict) or payload.get("status") not in statuses:
            raise RuntimeError(f"{tool_name} did not return a successful result: {payload}")
        csv_path = str(payload.get(path_field) or "").strip()
        if not csv_path:
            raise RuntimeError(f"{tool_name} did not return {path_field}.")
        path = Path(csv_path).expanduser()
        if not path.is_file():
            raise RuntimeError(f"{tool_name} returned a missing CSV path: {csv_path}")
        payload[path_field] = str(path.resolve())
        return payload

    def _format_mcp_failure(
        self,
        tool_name: str,
        exc: Exception,
        *,
        service_label: str,
        recovery: str,
    ) -> str:
        unavailable = getattr(self.mcp_client, "unavailable_servers", {}) or {}
        unavailable_text = (
            "; ".join(f"{name}: {reason}" for name, reason in unavailable.items())
            if unavailable
            else "none recorded"
        )
        return (
            f"{service_label} tool {tool_name!r} failed: {type(exc).__name__}: {exc}. "
            f"Unavailable MCP servers: {unavailable_text}. {recovery}"
        )

    def _resolve_candidate_csv_path_for_admet(
        self,
        *,
        candidate_csv_path: Optional[str],
    ) -> Optional[str]:
        explicit_path = str(candidate_csv_path or "").strip()
        if explicit_path:
            return str(self._validate_admet_candidate_csv_path(explicit_path))
        return None

    def _validate_admet_candidate_csv_path(self, candidate_csv_path: str) -> Path:
        text = str(candidate_csv_path or "").strip()
        path = Path(text).expanduser()
        if text in DISALLOWED_ADMET_CSV_PLACEHOLDERS or self._is_system_tmp_candidates_csv(path):
            raise ValueError(
                "Do not pass a candidates.csv placeholder to "
                "admet_filter_by_admetai. Use the exact candidate_csv_path from "
                "the preceding pipeline stage."
            )
        if path.is_dir():
            raise ValueError(f"Candidate CSV path is a directory, not a file: {text}")
        if not path.is_file():
            raise ValueError(f"Candidate CSV path does not exist: {text}")
        return path.resolve()

    @staticmethod
    def _is_system_tmp_candidates_csv(path: Path) -> bool:
        return (
            path.is_absolute()
            and path.name == "candidates.csv"
            and len(path.parts) >= 3
            and path.parts[-2] == "tmp"
        )

    def _normalize_admet_preference_json(self, preference_json: str) -> str:
        try:
            preferences = json.loads(preference_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("preference_json must be a JSON object string.") from exc
        if not isinstance(preferences, dict):
            raise ValueError("preference_json must decode to a JSON object.")

        property_preferences = self._property_preference_lookup()
        ambiguous_values = {"intermediate", "context_specific", "context-specific", "context specific"}
        normalized = {}
        invalid = {}
        ambiguous = {}
        for prop, direction in preferences.items():
            key = str(prop).strip()
            if not key:
                continue
            raw_value = str(direction).strip()
            raw_normalized = raw_value.casefold()
            meta_preference = property_preferences.get(key, "").casefold()
            if raw_normalized in ambiguous_values or (
                meta_preference in {"intermediate", "context_specific"}
                and raw_normalized == meta_preference
            ):
                ambiguous[key] = raw_value
                continue
            if raw_normalized not in {"higher", "lower"}:
                invalid[key] = raw_value
                continue
            value = raw_normalized
            normalized[key] = value
        if ambiguous:
            ambiguous_text = ", ".join(f"{key}={value!r}" for key, value in ambiguous.items())
            raise ValueError(
                "Ambiguous preference_json direction(s): "
                f"{ambiguous_text}. Intermediate/Context_Specific property preferences "
                "cannot be passed directly to admet_filter_by_admetai. Choose an explicit "
                "\"higher\" or \"lower\" direction only if the user goal makes it clear, "
                "otherwise omit that property from the filter. Example: {\"BBB_Martins\": \"higher\"}."
            )
        if invalid:
            invalid_text = ", ".join(f"{key}={value!r}" for key, value in invalid.items())
            raise ValueError(
                "Invalid preference_json direction(s): "
                f"{invalid_text}. Re-call admet_filter_by_admetai with a JSON object "
                "whose values are exactly \"higher\" or \"lower\". "
                "Example: {\"BBB_Martins\": \"higher\"}. "
                "Do not use forms such as higher_better/lower_better in the tool call."
            )
        if not normalized:
            raise ValueError(
                "preference_json must contain at least one property direction. "
                "Use exactly \"higher\" or \"lower\" as values, for example "
                "{\"BBB_Martins\": \"higher\"}."
            )
        return enforce_frozen_admet_preference_json(
            canonical_admet_preference_json(normalized),
            self.event_context.get("selected_admet_preference_json"),
            tool_name="admet_filter_by_admetai",
        )

    def _property_preference_lookup(self) -> Dict[str, str]:
        try:
            data = json.loads(self._load_property_meta_info())
        except Exception:
            return {}
        lookup: Dict[str, str] = {}
        if not isinstance(data, dict):
            return lookup
        for category in data.values():
            if not isinstance(category, dict):
                continue
            for prop, meta in category.items():
                if isinstance(meta, dict):
                    preference = meta.get("Preference")
                    if preference is not None:
                        lookup[str(prop)] = str(preference)
        return lookup
    
    def _load_property_meta_info(self) -> str:
        """Load property meta info from file.
        
        Returns:
            Property meta info as JSON string
        """
        path = self.config.paths.property_meta_info_path
        if not path:
            return "{}"
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "{}"
        except Exception as e:
            return f'{{"error": "{str(e)}"}}'


async def create_creative_molecule_explorer_agent(
    config: AgentConfig,
    mcp_client: MCPClientManager,
    additional_context: Optional[str] = None,
    medchem_retrieval_agent: Optional[Any] = None,
) -> CreativeMoleculeExplorerAgent:
    """Factory function to create a creative molecule explorer agent.
    
    Args:
        config: Agent configuration
        mcp_client: MCP client manager
        additional_context: Optional user-provided context to inject at system prompt start
        
    Returns:
        Configured agent instance
    """
    agent = CreativeMoleculeExplorerAgent(
        config,
        mcp_client,
        additional_context,
        medchem_retrieval_agent=medchem_retrieval_agent,
    )
    await agent.build()
    return agent
