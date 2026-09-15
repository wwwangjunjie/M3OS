"""Rational Medicinal Designer agent.

This agent simulates the thought process of human medicinal chemists,
focusing on logical interpretability, medicinal chemistry knowledge,
and case-informed design strategy.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.tools import tool

from m3os.agents_v5.agents.base import BaseAgent
from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.services.mcp_client import (
    MCPClientManager,
    PROTEIN_CONTEXT_TOOL_NAMES,
    TOOL_NAMES_RATIONAL,
    filter_tools_by_name,
)
from m3os.agents_v5.services.smiles_validation import validate_smiles
from m3os.agents_v5.prompts.system_loader import render_rational_system_prompt


_COMPLEX_STRUCTURE_SUFFIXES = {".pdb", ".ent", ".cif", ".mmcif"}


class RationalMedicinalDesignerAgent(BaseAgent):
    """Agent for rational medicinal chemistry design.
    
    This agent simulates the thought process of human medicinal chemists:
    1. Retrieve medicinal chemistry knowledge and relevant optimization cases
    2. Develop precise modification strategies based on theory, SAR, and cases
    3. Manually construct and validate each optimized SMILES with detailed rationale
    """
    
    def __init__(
        self,
        config: AgentConfig,
        mcp_client: MCPClientManager,
        additional_context: Optional[str] = None,
        medchem_retrieval_agent: Optional[Any] = None,
    ):
        """Initialize the rational medicinal designer agent.
        
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
        self._initial_boltz_identity: Optional[tuple[str, str]] = None
        self._initial_boltz_structure_path: Optional[str] = None
        self._initial_boltz_result: Any = None
        self._initial_plip_results: Dict[str, tuple[tuple[str, str], Any]] = {}
    
    @property
    def system_prompt(self) -> str:
        """Get the system prompt for this agent.
        
        Returns:
            Formatted system prompt with knowledge
        """
        return render_rational_system_prompt(
            additional_context=self.additional_context,
        )
    
    @property
    def output_schema(self) -> type | None:
        """Get the output schema for this agent.
        
        Returns:
            None because the rational agent now writes free-form reasoning.
        """
        return None

    @property
    def agent_role(self) -> str:
        return "rational"

    def _tool_description_for_wrapping(self, tool_name: str, description: str) -> str:
        if tool_name == "query_evolutionary_optimization_cases":
            routing_note = (
                "Rational Designer evidence note: use this to retrieve historical "
                "molecular evolution subgraphs, then integrate the case signal "
                "with medicinal chemistry knowledge before proposing molecules. "
                "Do not treat case retrieval alone as sufficient design reasoning."
            )
        else:
            return super()._tool_description_for_wrapping(tool_name, description)
        if not description:
            return routing_note
        return f"{routing_note}\n\n{description}"
    
    async def get_tools(self) -> List[Any]:
        """Get tools for the rational medicinal designer.
        
        Returns:
            List of tools for knowledge-grounded, case-informed reasoning
        """
        if self._tools is None:
            all_tools = self.mcp_client.get_all_agent_tools()
            direct_tool_names = TOOL_NAMES_RATIONAL - PROTEIN_CONTEXT_TOOL_NAMES
            self._tools = [
                *filter_tools_by_name(all_tools, direct_tool_names),
            ]
            if self.config.enable_molecular_auxiliary_context:
                available_names = {
                    str(getattr(tool_item, "name", "") or "")
                    for tool_item in all_tools
                }
                if "generate_protein_ligand_complex" in available_names:
                    self._tools.append(self._build_initial_complex_generation_tool())
                if "analyze_protein_ligand_interactions" in available_names:
                    self._tools.append(self._build_initial_interaction_analysis_tool())
            if self.medchem_retrieval_agent is not None:
                self._tools.append(self._build_ask_medchem_knowledge_tool())

        return self._tools

    def _build_initial_complex_generation_tool(self) -> Any:
        @tool("generate_protein_ligand_complex")
        async def generate_protein_ligand_complex_tool(data: str) -> str:
            """Generate a protein-ligand complex structure from a JSON string.

            The JSON object contains `sequence` and ligand `smiles`. The result
            contains the generated `structure_path` when successful.
            """
            self._require_initial_structure_round("Boltz")
            uploaded_paths = self._uploaded_complex_structure_paths()
            if uploaded_paths:
                raise ValueError(
                    "Boltz is not allowed because an uploaded structure is available. "
                    "Call analyze_protein_ligand_interactions with its exact path."
                )

            request = self._parse_json_object(data, "data")
            requested_sequence = self._normalize_protein_sequence(request.get("sequence"))
            initial_sequence = self._normalize_protein_sequence(
                self.event_context.get("initial_protein_sequence")
            )
            if not initial_sequence:
                raise ValueError(
                    "Boltz requires the target protein sequence from the task input."
                )
            if requested_sequence != initial_sequence:
                raise ValueError(
                    "Boltz is restricted to the task's exact initial target protein sequence."
                )

            requested_smiles = self._canonical_smiles(request.get("smiles"), "requested")
            initial_smiles = self._canonical_smiles(
                self.event_context.get("initial_smiles"),
                "initial",
            )
            if requested_smiles != initial_smiles:
                raise ValueError(
                    "Boltz is restricted to the task's exact initial ligand; generated, "
                    "optimized, current, and batch candidates are not allowed."
                )

            identity = (initial_smiles, initial_sequence)
            if self._initial_boltz_identity == identity and self._initial_boltz_result is not None:
                return self._initial_boltz_result

            result = await self.mcp_client.invoke_tool(
                "generate_protein_ligand_complex",
                {"data": json.dumps(request, ensure_ascii=False)},
            )
            result_payload = self._parse_tool_result_object(result)
            structure_path = str(result_payload.get("structure_path") or "").strip()
            if result_payload.get("status") != "success" or not structure_path:
                raise RuntimeError(
                    "Boltz did not return status='success' with a structure_path."
                )
            self._initial_boltz_identity = identity
            self._initial_boltz_structure_path = self._normalize_structure_path(
                structure_path
            )
            self._initial_boltz_result = result
            return result

        return generate_protein_ligand_complex_tool

    def _build_initial_interaction_analysis_tool(self) -> Any:
        @tool("analyze_protein_ligand_interactions")
        async def analyze_protein_ligand_interactions_tool(structure_path: str) -> str:
            """Extract protein-ligand interactions from a complex structure file.

            `structure_path` accepts a supported complex structure path and the
            result groups detected contacts by interaction type.
            """
            self._require_initial_structure_round("PLIP")
            normalized_path = self._normalize_structure_path(structure_path)
            allowed_paths = self._uploaded_complex_structure_paths()
            if (
                self._initial_boltz_structure_path
                and self._initial_boltz_identity == self._current_task_identity()
            ):
                allowed_paths.add(self._initial_boltz_structure_path)
            if normalized_path not in allowed_paths:
                raise ValueError(
                    "PLIP is restricted to an exact uploaded initial-complex path or "
                    "the structure_path returned by the allowed initial Boltz call."
                )
            task_identity = self._current_task_identity()
            cached_result = self._initial_plip_results.get(normalized_path)
            if cached_result and cached_result[0] == task_identity:
                return cached_result[1]

            result = await self.mcp_client.invoke_tool(
                "analyze_protein_ligand_interactions",
                {"structure_path": structure_path},
            )
            self._initial_plip_results[normalized_path] = (task_identity, result)
            return result

        return analyze_protein_ligand_interactions_tool

    def _require_initial_structure_round(self, tool_label: str) -> None:
        round_value = self.event_context.get("mcgs_round")
        if round_value is None:
            round_value = self.event_context.get("round")
        try:
            is_initial_round = int(round_value) == 1
        except (TypeError, ValueError):
            is_initial_round = False
        if not is_initial_round:
            raise ValueError(
                f"{tool_label} is available only in the first Rational MCGS round. "
                "Reuse the initial PLIP interaction text for later reasoning."
            )

    def get_initial_interaction_evidence(self, max_chars: int = 12000) -> str:
        """Return cached initial PLIP evidence for later text-only reasoning."""
        if not self._initial_plip_results:
            return ""
        task_identity = self._current_task_identity()
        evidence_items: List[Any] = []
        for cached_identity, result in self._initial_plip_results.values():
            if cached_identity != task_identity:
                continue
            try:
                evidence_items.append(self._parse_json_object(result, "PLIP result"))
            except ValueError:
                evidence_items.append(str(result))
        if not evidence_items:
            return ""
        evidence = json.dumps(evidence_items, ensure_ascii=False, default=str)
        if len(evidence) <= max_chars:
            return evidence
        return evidence[:max_chars].rstrip() + "...[initial PLIP evidence shortened]"

    def _current_task_identity(self) -> tuple[str, str]:
        smiles = str(self.event_context.get("initial_smiles") or "").strip()
        if smiles:
            valid, canonical, _error = validate_smiles(smiles)
            if valid:
                smiles = canonical
        sequence = self._normalize_protein_sequence(
            self.event_context.get("initial_protein_sequence")
        )
        return smiles, sequence

    def _uploaded_complex_structure_paths(self) -> set[str]:
        context = str(self.additional_context or "")
        marker = "[USER-UPLOADED STRUCTURE FILE CONTEXT]"
        if marker not in context:
            return set()
        paths: set[str] = set()
        for match in re.finditer(r"(?m)^Path:\s*(.+?)\s*$", context):
            candidate = match.group(1).strip().strip('"\'')
            if Path(candidate).suffix.lower() in _COMPLEX_STRUCTURE_SUFFIXES:
                paths.add(self._normalize_structure_path(candidate))
        return paths

    @staticmethod
    def _normalize_structure_path(structure_path: Any) -> str:
        text = str(structure_path or "").strip()
        if not text:
            raise ValueError("structure_path must be a non-empty path.")
        return str(Path(text).expanduser().resolve(strict=False))

    @staticmethod
    def _normalize_protein_sequence(sequence: Any) -> str:
        if not isinstance(sequence, str):
            return ""
        residues = [
            line.strip()
            for line in sequence.splitlines()
            if line.strip() and not line.lstrip().startswith(">")
        ]
        return re.sub(r"\s+", "", "".join(residues)).upper()

    @staticmethod
    def _canonical_smiles(smiles: Any, label: str) -> str:
        valid, canonical, error = validate_smiles(smiles)
        if not valid:
            raise ValueError(f"The {label} ligand SMILES is invalid: {error}")
        return canonical

    @staticmethod
    def _parse_json_object(value: Any, label: str) -> Dict[str, Any]:
        if isinstance(value, dict):
            parsed = value
        else:
            try:
                parsed = json.loads(str(value or ""))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} must be a valid JSON object string.") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} must decode to a JSON object.")
        return parsed

    @classmethod
    def _parse_tool_result_object(cls, result: Any) -> Dict[str, Any]:
        try:
            return cls._parse_json_object(result, "Boltz result")
        except ValueError as exc:
            raise RuntimeError("Boltz returned a non-JSON result.") from exc


async def create_rational_medicinal_designer_agent(
    config: AgentConfig,
    mcp_client: MCPClientManager,
    additional_context: Optional[str] = None,
    medchem_retrieval_agent: Optional[Any] = None,
) -> RationalMedicinalDesignerAgent:
    """Factory function to create a rational medicinal designer agent.
    
    Args:
        config: Agent configuration
        mcp_client: MCP client manager
        additional_context: Optional user-provided context to inject at system prompt start
        
    Returns:
        Configured agent instance
    """
    agent = RationalMedicinalDesignerAgent(
        config,
        mcp_client,
        additional_context,
        medchem_retrieval_agent=medchem_retrieval_agent,
    )
    await agent.build()
    return agent
