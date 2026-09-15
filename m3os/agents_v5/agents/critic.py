"""Critic agent for molecular evaluation.

This agent acts as a 'Quality Gate' for molecular optimization,
evaluating candidate molecules by integrating quantitative ADMET metrics
with qualitative medicinal chemistry expertise.
"""

import ast
import json
from pathlib import Path
import re
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

from langchain.tools import tool

from m3os.agents_v5.agents.base import BaseAgent
from m3os.agents_v5.core.config import AgentConfig, PROJECT_TMP_DIR
from m3os.agents_v5.services.mcp_client import (
    MCPClientManager,
    TOOL_NAMES_CRITIC,
    filter_tools_by_name,
)
from m3os.agents_v5.prompts.system_loader import render_critic_system_prompt
from m3os.agents_v5.services.smiles_validation import validate_smiles


NESSO_CRITIC_BATCH_SIZE = 4
NESSO_AFFINITY_GATE_EPSILON = 1e-9


class CriticAgent(BaseAgent):
    """Agent for critically evaluating molecular optimizations.
    
    This agent acts as a Lead Computational Medicinal Chemist:
    1. Integrates quantitative ADMET metrics with qualitative expertise
    2. Performs structural sanity checks (toxicophores, PAINS, synthetic accessibility)
    3. Ensures pharmacophore integrity relative to parent molecule
    4. Determines the 'Reward' for Monte Carlo Graph Search (MCGS)
    """
    
    def __init__(
        self,
        config: AgentConfig,
        mcp_client: MCPClientManager,
        additional_context: Optional[str] = None,
        medchem_retrieval_agent: Optional[Any] = None,
    ):
        """Initialize the critic agent.
        
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
        self._nesso_prediction_cache: Dict[tuple[str, tuple[str, ...]], str] = {}
    
    @property
    def system_prompt(self) -> str:
        """Get the system prompt for this agent.
        
        Returns:
            Formatted system prompt with knowledge and property meta info
        """
        # Load property meta info
        property_meta_info = self._load_property_meta_info()
        
        return render_critic_system_prompt(
            property_meta_info=property_meta_info,
            additional_context=self.additional_context,
        )
    
    @property
    def output_schema(self) -> type | None:
        """Get the output schema for this agent.
        
        Returns:
            None because the critic now writes free-form evaluation.
        """
        return None

    @property
    def agent_role(self) -> str:
        return "critic"

    def _tool_description_for_wrapping(self, tool_name: str, description: str) -> str:
        return super()._tool_description_for_wrapping(tool_name, description)
    
    async def get_tools(self) -> List[Any]:
        """Get tools for the critic agent.
        
        Returns:
            List of tools for ADMET prediction
        """
        if self._tools is None:
            # Get all tools from MCP servers
            all_tools = self.mcp_client.get_all_agent_tools()

            # Filter to only critic tools
            direct_tool_names = TOOL_NAMES_CRITIC - {"evaluate_nesso_activity"}
            self._tools = [*filter_tools_by_name(all_tools, direct_tool_names)]
            if self.medchem_retrieval_agent is not None:
                self._tools.append(self._build_ask_medchem_knowledge_tool())
            self._tools.append(self._build_candidate_constraints_tool())
            self._tools.append(self._build_nesso_activity_tool())
        
        return self._tools

    def _build_candidate_constraints_tool(self) -> Any:
        @tool("evaluate_candidate_constraints")
        async def evaluate_candidate_constraints_tool(smiles_list: str) -> str:
            """Evaluate candidates against a complete, explicit runtime contract.

            Call this only when the attached contract contains at least one
            directly evaluable hard, interval, hold, or baseline-relative
            constraint. A property with only an increase/decrease direction is
            a monotonic optimization preference, not a constraint; when no
            evaluable constraint exists, skip this tool and assess the available
            predictions under the ordinary optimization objective instead.
            Never invent missing sources, endpoints, operators, baselines,
            tolerances, or thresholds.
            """
            smiles = self._parse_smiles_list(smiles_list)
            contract = self.event_context.get("leadopt_task_contract")
            if not isinstance(contract, dict) or not contract:
                raise ValueError(
                    "Candidate constraint evaluation requires a lead-optimization task contract."
                )
            reference_smiles = str(
                contract.get("reference_smiles")
                or self.event_context.get("initial_smiles")
                or ""
            ).strip()
            if not reference_smiles:
                raise ValueError("Candidate constraint evaluation requires an initial molecule.")
            input_dir = PROJECT_TMP_DIR / "critic_constraint_inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            input_csv = (input_dir / f"critic_candidates_{uuid.uuid4().hex}.csv").resolve()
            pd.DataFrame({"SMILES": smiles}).to_csv(input_csv, index=False)
            try:
                result = await self.mcp_client.invoke_tool(
                    "admet_filter_by_task_constraints",
                    {
                        "task_contract_json": json.dumps(contract, ensure_ascii=False),
                        "reference_smiles": reference_smiles,
                        "candidate_csv_path": str(input_csv),
                    },
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Candidate constraint evaluation failed: {type(exc).__name__}: {exc}"
                ) from exc
            payload = self._parse_json_object(result, "candidate constraints")
            scored_path = Path(str(payload.get("scored_csv_path") or "")).expanduser()
            if not scored_path.is_file():
                raise RuntimeError("Candidate constraint evaluation returned no scored CSV.")
            records = json.loads(pd.read_csv(scored_path).to_json(orient="records"))
            return json.dumps(
                {
                    "status": "success",
                    "stage": "candidate_constraints",
                    "input_count": int(payload.get("input_count") or len(smiles)),
                    "pass_count": int(payload.get("output_count") or 0),
                    "records": records,
                },
                ensure_ascii=False,
            )

        return evaluate_candidate_constraints_tool

    def _build_nesso_activity_tool(self) -> Any:
        @tool("evaluate_nesso_activity")
        async def evaluate_nesso_activity_tool(
            smiles_list: str,
            protein_sequence: str = "",
            protein_id: str = "target",
        ) -> str:
            """Evaluate and rank candidates using both Nesso activity outputs.

            M3OS binds the authoritative target protein and adds the initial
            and current molecules as comparison baselines. The result contains
            Nesso binding probabilities, affinity values, task-gate flags, and
            a deterministic Nesso ranking.
            """
            del protein_sequence
            smiles = self._parse_smiles_list(smiles_list)
            resolved_protein_id = str(protein_id or "target").strip() or "target"
            comparison_smiles = self._comparison_smiles()
            all_smiles = list(dict.fromkeys([*smiles, *comparison_smiles]))
            nesso = await self._predict_nesso(
                all_smiles,
                protein_id=resolved_protein_id,
            )
            return json.dumps(
                self._build_nesso_activity_result(
                    requested_smiles=smiles,
                    comparison_smiles=comparison_smiles,
                    nesso_records=nesso,
                ),
                ensure_ascii=False,
            )

        return evaluate_nesso_activity_tool

    def _protein_input_params(self) -> Dict[str, str]:
        fasta_value = str(self.event_context.get("protein_fasta_path") or "").strip()
        if fasta_value:
            fasta = Path(fasta_value).expanduser().resolve()
            if not fasta.is_file():
                raise ValueError(f"Task protein FASTA does not exist: {fasta}")
            return {"protein_fasta_path": str(fasta)}
        sequence = self._task_protein_sequence()
        if not sequence:
            raise ValueError("Nesso activity evaluation requires a target protein input.")
        return {"protein_sequence": sequence}

    def _protein_cache_identity(self) -> str:
        params = self._protein_input_params()
        return str(params.get("protein_fasta_path") or params.get("protein_sequence") or "")

    def _comparison_smiles(self) -> List[str]:
        values = [
            str(self.event_context.get("initial_smiles") or "").strip(),
            str(self.event_context.get("current_smiles") or "").strip(),
        ]
        return list(dict.fromkeys(value for value in values if value))

    async def _predict_nesso(
        self,
        smiles: List[str],
        *,
        protein_id: str,
    ) -> List[Dict[str, Any]]:
        cache_key = (self._protein_cache_identity(), tuple(smiles))
        cached = self._nesso_prediction_cache.get(cache_key)
        if cached is not None:
            return self._parse_prediction_records(cached, "Nesso")
        combined: List[Dict[str, Any]] = []
        for offset in range(0, len(smiles), NESSO_CRITIC_BATCH_SIZE):
            batch = smiles[offset : offset + NESSO_CRITIC_BATCH_SIZE]
            try:
                result = await self.mcp_client.invoke_tool(
                    "nesso_predict_by_cofolding",
                    {
                        "smiles_list": json.dumps(batch, ensure_ascii=False),
                        "protein_id": protein_id,
                        **self._protein_input_params(),
                    },
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Nesso activity prediction failed: {type(exc).__name__}: {exc}"
                ) from exc
            records = self._parse_prediction_records(result, "Nesso")
            if len(records) != len(batch):
                raise RuntimeError("Nesso returned an incomplete prediction set.")
            combined.extend(records)
        self._nesso_prediction_cache[cache_key] = str(combined)
        return combined

    def _task_protein_sequence(self) -> str:
        raw_sequence = str(self.event_context.get("initial_protein_sequence") or "")
        residues = []
        for line in raw_sequence.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(">"):
                residues.append("".join(stripped.split()))
        return "".join(residues).upper()

    @staticmethod
    def _canonical_smiles_key(smiles: Any) -> str:
        valid, canonical, _error = validate_smiles(str(smiles or "").strip())
        return canonical if valid and canonical else str(smiles or "").strip()

    def _activity_probability_threshold(self) -> float:
        contract = self.event_context.get("leadopt_task_contract")
        if not isinstance(contract, dict):
            return 0.0
        for entry in contract.get("hard_constraints") or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("property") or "").casefold() != "boltz_binding_probability":
                continue
            match = re.search(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", str(entry.get("threshold") or ""))
            if match:
                return float(match.group(0))
        return 0.0

    def _nesso_affinity_task_gate(
        self,
        candidate_affinity: float,
        reference_affinity: float,
    ) -> bool:
        contract = self.event_context.get("leadopt_task_contract")
        if not isinstance(contract, dict):
            return True
        objectives = contract.get("optimization_objectives", contract.get("objectives", []))
        holds = contract.get("hold_constant", contract.get("holds", []))
        for entry in objectives or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("property") or "").casefold() != "binding_affinity":
                continue
            threshold = float(entry.get("threshold", 0.0))
            direction = str(entry.get("direction") or "minimize").casefold()
            if direction == "maximize":
                improvement = candidate_affinity - reference_affinity
            else:
                improvement = reference_affinity - candidate_affinity
            return improvement + NESSO_AFFINITY_GATE_EPSILON >= threshold
        for entry in holds or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("property") or "").casefold() != "binding_affinity":
                continue
            tolerance = float(entry.get("tolerance", 0.0))
            direction = str(entry.get("direction") or "change").casefold()
            delta = candidate_affinity - reference_affinity
            if direction == "increase":
                return delta <= tolerance
            if direction == "decrease":
                return -delta <= tolerance
            return abs(delta) <= tolerance
        return True

    @staticmethod
    def _rank_percentiles(values: Dict[str, float], *, higher_is_better: bool) -> Dict[str, float]:
        ordered = sorted(values, key=values.get, reverse=higher_is_better)
        denominator = max(len(ordered) - 1, 1)
        return {
            key: 1.0 - (index / denominator)
            for index, key in enumerate(ordered)
        }

    def _build_nesso_activity_result(
        self,
        *,
        requested_smiles: List[str],
        comparison_smiles: List[str],
        nesso_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        nesso_by_key = {
            self._canonical_smiles_key(record.get("SMILES")): record
            for record in nesso_records
        }
        reference_smiles = (
            str(self.event_context.get("initial_smiles") or "").strip()
            or (comparison_smiles[0] if comparison_smiles else "")
        )
        reference_key = self._canonical_smiles_key(reference_smiles)
        reference_nesso = nesso_by_key.get(reference_key)
        if not reference_nesso:
            raise RuntimeError("Nesso activity evaluation requires the initial molecule.")

        reference_nesso_probability = float(
            reference_nesso["Nesso_affinity_probability_binary"]
        )
        reference_nesso_affinity = float(reference_nesso["Nesso_affinity_pred_value"])
        threshold = self._activity_probability_threshold()
        rows: list[dict[str, Any]] = []
        for smiles in requested_smiles:
            key = self._canonical_smiles_key(smiles)
            nesso = nesso_by_key.get(key)
            if not nesso:
                raise RuntimeError(f"Missing Nesso prediction for candidate {smiles}")
            nesso_probability = float(nesso["Nesso_affinity_probability_binary"])
            nesso_affinity = float(nesso["Nesso_affinity_pred_value"])
            nesso_pass = nesso_probability > threshold
            nesso_affinity_task_pass = self._nesso_affinity_task_gate(
                nesso_affinity,
                reference_nesso_affinity,
            )
            rows.append(
                {
                    "SMILES": key,
                    "Nesso_affinity_probability_binary": nesso_probability,
                    "Nesso_affinity_pred_value": nesso_affinity,
                    "Nesso_probability_task_pass": nesso_pass,
                    "Nesso_affinity_improved_vs_initial": (
                        nesso_affinity < reference_nesso_affinity
                    ),
                    "Nesso_affinity_improvement_vs_initial": round(
                        reference_nesso_affinity - nesso_affinity,
                        6,
                    ),
                    "Nesso_affinity_task_pass": nesso_affinity_task_pass,
                    "Nesso_activity_pass": bool(
                        nesso_pass and nesso_affinity_task_pass
                    ),
                }
            )

        nesso_affinity_values = {
            row["SMILES"]: row["Nesso_affinity_pred_value"]
            for row in rows
        }
        nesso_affinity_percentiles = self._rank_percentiles(
            nesso_affinity_values,
            higher_is_better=False,
        )
        for row in rows:
            key = row["SMILES"]
            row["Nesso_activity_score"] = round(
                nesso_affinity_percentiles[key],
                6,
            )
        rows.sort(
            key=lambda row: (
                bool(row["Nesso_activity_pass"]),
                float(row["Nesso_activity_score"]),
                float(row["Nesso_affinity_probability_binary"]),
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows, start=1):
            row["Nesso_activity_rank"] = rank
        return {
            "status": "success",
            "stage": "nesso_activity_evaluation",
            "probability_proxy_threshold": threshold,
            "probability_requires_better_than_initial": False,
            "ranking_metric": "Nesso_affinity_improvement_vs_initial",
            "initial_baseline": {
                "SMILES": reference_key,
                "Nesso_affinity_probability_binary": reference_nesso_probability,
                "Nesso_affinity_pred_value": reference_nesso_affinity,
            },
            "records": rows,
        }

    @staticmethod
    def _parse_smiles_list(smiles_list: str) -> List[str]:
        text = str(smiles_list or "").strip()
        parsed: Any = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                break
            except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                continue
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("smiles_list must be a Python/JSON list of SMILES strings.")
        smiles = [str(item).strip() for item in parsed if str(item).strip()]
        if not smiles:
            raise ValueError("smiles_list must contain at least one SMILES string.")
        return smiles

    @staticmethod
    def _parse_json_object(result: Any, label: str) -> Dict[str, Any]:
        if isinstance(result, dict):
            payload = result
        else:
            try:
                payload = json.loads(str(result or ""))
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"{label} returned invalid JSON.") from exc
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise RuntimeError(f"{label} returned an unsuccessful result: {payload}")
        return dict(payload)

    @staticmethod
    def _parse_prediction_records(result: Any, model_name: str) -> List[Dict[str, Any]]:
        if isinstance(result, list):
            parsed = result
        else:
            text = str(result or "").strip()
            parsed: Any = None
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    break
                except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
                    continue
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            raise RuntimeError(f"{model_name} returned an invalid prediction record list.")
        return [dict(item) for item in parsed]

    @staticmethod
    def _parse_nesso_records(result: Any) -> List[Dict[str, Any]]:
        return CriticAgent._parse_prediction_records(result, "Nesso")
    
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
    
async def create_critic_agent(
    config: AgentConfig,
    mcp_client: MCPClientManager,
    additional_context: Optional[str] = None,
    medchem_retrieval_agent: Optional[Any] = None,
) -> CriticAgent:
    """Factory function to create a critic agent.
    
    Args:
        config: Agent configuration
        mcp_client: MCP client manager
        additional_context: Optional user-provided context to inject at system prompt start
        
    Returns:
        Configured agent instance
    """
    agent = CriticAgent(
        config,
        mcp_client,
        additional_context,
        medchem_retrieval_agent=medchem_retrieval_agent,
    )
    await agent.build()
    return agent
