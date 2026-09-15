"""HTML report generation utilities for agents_v5 optimization results."""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, rdFMCS, rdMolDescriptors

from m3os.agents_v5.agents.report import ReportAgent
from m3os.agents_v5.core.config import PROJECT_ROOT, PROJECT_TMP_DIR
from m3os.agents_v5.core.models import (
    OptimizationReport,
    ReportBlueprint,
    ReportDesignedMolecule,
    ReportFirstRoundCandidate,
    ReportModificationSite,
    ReportStrategyGroup,
)
from m3os.agents_v5.core.state import MCGSState
from m3os.agents_v5.services.smiles_validation import parse_smiles_quiet


PROPERTY_COLUMNS = ["MW", "cLogP", "TPSA", "HBD", "HBA", "ArR", "RotB", "Fsp3", "Ro5"]
REPORT_MCS_TIMEOUT_SECONDS = 2
MAX_AUTO_DIFF_GALLERY_IMAGES = 24
COMMON_HIGHLIGHT_COLOR = (0.38, 0.70, 0.42)
CHANGE_HIGHLIGHT_COLOR = (0.93, 0.48, 0.18)
REPORT_ASSET_IMAGE_TMP_SUBDIR = Path("report_assets") / "images"
REPORT_SMILES_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]{3,240})(?![A-Za-z0-9])"
)


@dataclass
class ReportResult:
    """Generated report payload."""

    html: str
    output_path: str
    narrative: Any
    mode: str = "fallback"
    fallback_reason: str = ""


class OptimizationReportGenerator:
    """Build a self-contained v3_2-style HTML SAR report from an MCGS state."""

    def __init__(self, report_agent: Optional[ReportAgent] = None):
        self.report_agent = report_agent

    async def generate(
        self,
        state: MCGSState,
        output_dir: str | Path = "tmp/reports",
        top_n: int = 8,
        first_round_n: int = 5,
        filename: Optional[str] = None,
        coverage_mode: str = "all",
        max_detailed_cards: Optional[int] = None,
        style_reference: str = "dark_medicinal_design_board",
        allow_fallback: bool = True,
        report_language: Optional[str] = None,
        report_request_text: str = "",
    ) -> ReportResult:
        mol_graph = state.get("mcgs_graph")
        candidate_nodes = self._extract_candidate_nodes(mol_graph)
        if not candidate_nodes:
            candidate_nodes = self._extract_fallback_candidates(state)
        candidate_nodes = self._select_report_candidates(
            self._dedupe_candidates(candidate_nodes),
            coverage_mode=coverage_mode,
            top_n=top_n,
        )
        graph_statistics = self._extract_graph_statistics(mol_graph)
        initial_smiles, initial_iupac, initial_fragments = _resolve_initial_report_facts(state, mol_graph)
        state_for_report = dict(state)
        state_for_report.update(
            {
                "initial_smiles": initial_smiles,
                "initial_iupac": initial_iupac,
                "initial_fragments": initial_fragments,
            }
        )
        effective_report_language = _resolve_report_language(
            report_language,
            report_request_text=report_request_text,
            state=state_for_report,
        )
        effective_max_detailed_cards = _resolve_max_detailed_cards(
            len(candidate_nodes),
            max_detailed_cards=max_detailed_cards,
            top_n=top_n,
            first_round_n=first_round_n,
        )
        effective_style_reference = str(style_reference or "dark_medicinal_design_board").strip()
        if effective_style_reference.lower() in {"agentic", "default", "html"}:
            effective_style_reference = "dark_medicinal_design_board"

        report_assets = self._build_report_assets(
            initial_smiles=initial_smiles,
            initial_iupac=initial_iupac,
            initial_fragments=initial_fragments,
            candidate_nodes=candidate_nodes,
            report_language=effective_report_language,
        )

        if self.report_agent is not None:
            try:
                blueprint = await self._build_report_blueprint(
                    state=state_for_report,
                    candidate_nodes=candidate_nodes,
                    graph_statistics=graph_statistics,
                    report_assets=report_assets,
                    coverage_mode=coverage_mode,
                    top_n=top_n,
                    first_round_n=first_round_n,
                    max_detailed_cards=effective_max_detailed_cards,
                    style_reference=effective_style_reference,
                    report_language=effective_report_language,
                )
                html_text = await self._build_agentic_html(
                    state=state_for_report,
                    blueprint=blueprint,
                    report_assets=report_assets,
                    candidate_nodes=candidate_nodes,
                    graph_statistics=graph_statistics,
                    coverage_mode=coverage_mode,
                    top_n=top_n,
                    max_detailed_cards=effective_max_detailed_cards,
                    style_reference=effective_style_reference,
                    report_language=effective_report_language,
                )
                html_text = prepare_agentic_report_html(
                    html_text,
                    report_assets=report_assets,
                    candidate_nodes=candidate_nodes,
                    report_language=effective_report_language,
                )
                output_path = self._write_html(
                    html_text=html_text,
                    output_dir=Path(output_dir),
                    filename=filename or self._default_filename(state_for_report, mol_graph),
                )
                return ReportResult(html=html_text, output_path=str(output_path), narrative=blueprint, mode="agentic")
            except Exception as exc:
                if not allow_fallback:
                    raise RuntimeError(f"ReportAgent failed to generate an agentic HTML report: {exc}") from exc
                print(f"[Report warning] Agentic reporter failed, using fallback report: {exc}")
        elif not allow_fallback:
            raise RuntimeError("ReportAgent is not available; cannot generate an agentic HTML report.")

        fallback = self._build_fallback_result(
            state=state_for_report,
            candidate_nodes=candidate_nodes,
            initial_smiles=initial_smiles,
            initial_iupac=initial_iupac,
            initial_fragments=initial_fragments,
            first_round_n=first_round_n,
            report_language=effective_report_language,
        )
        output_path = self._write_html(
            html_text=fallback.html,
            output_dir=Path(output_dir),
            filename=filename or self._default_filename(state_for_report, mol_graph),
        )
        return ReportResult(
            html=fallback.html,
            output_path=str(output_path),
            narrative=fallback.narrative,
            mode="fallback",
            fallback_reason="report_agent_failed_or_unavailable",
        )

    async def _build_report_content(
        self,
        *,
        state: MCGSState,
        candidate_nodes: Sequence[Dict[str, Any]],
        graph_statistics: Dict[str, Any],
    ) -> OptimizationReport:
        if self.report_agent is None:
            return build_fallback_report(
                initial_smiles=state.get("initial_smiles", ""),
                optimization_goal=state.get("optimization_goal", ""),
                candidate_nodes=candidate_nodes,
            )

        try:
            query = self.report_agent.format_user_prompt(
                user_prompt=state.get("user_prompt", ""),
                optimization_goal=state.get("optimization_goal", ""),
                project_manager_brief=state.get("project_manager_brief", ""),
                initial_smiles=state.get("initial_smiles", ""),
                initial_iupac=state.get("initial_iupac", ""),
                initial_fragments=state.get("initial_fragments", ""),
                graph_statistics=graph_statistics,
                candidate_nodes=list(candidate_nodes),
                current_info_list=state.get("current_info_list", []),
            )
            result = await self.report_agent.invoke(query)
            structured = result.get("structured_response") if isinstance(result, dict) else result
            report = _coerce_report(structured)
            if report is not None:
                return report
        except Exception as exc:
            print(f"[Report warning] ReportAgent failed, using fallback report: {exc}")

        return build_fallback_report(
            initial_smiles=state.get("initial_smiles", ""),
            optimization_goal=state.get("optimization_goal", ""),
            candidate_nodes=candidate_nodes,
        )

    def _select_report_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        coverage_mode: str,
        top_n: int,
    ) -> List[Dict[str, Any]]:
        normalized_mode = str(coverage_mode or "all").strip().lower()
        valid_candidates = [item for item in candidates if item.get("smiles")]
        if normalized_mode == "all":
            return list(valid_candidates)
        if top_n <= 0:
            return list(valid_candidates)
        return list(valid_candidates)[:top_n]

    def _build_report_assets(
        self,
        *,
        initial_smiles: str,
        initial_iupac: Any,
        initial_fragments: Any,
        candidate_nodes: Sequence[Dict[str, Any]],
        report_language: str = "zh",
    ) -> Dict[str, Any]:
        parent_props = calculate_molecule_properties(initial_smiles)
        parent_image = molecule_png_base64(
            initial_smiles,
            size=(520, 416),
            asset_name="parent_topology",
        )
        candidate_assets = []
        for index, node in enumerate(candidate_nodes, start=1):
            smiles = str(node.get("smiles") or "").strip()
            asset_id = f"M{index:03d}"
            props = calculate_molecule_properties(smiles)
            candidate_assets.append(
                {
                    "asset_id": asset_id,
                    "image_placeholder": f"{{{{image:{asset_id}}}}}",
                    "topology_image_placeholder": f"{{{{image:{asset_id}}}}}",
                    "diff_image_placeholder": f"{{{{image_diff:{asset_id}}}}}",
                    "diff_highlight_legend": (
                        "RDKit MCS comparison: green marks the shared initial/candidate scaffold; "
                        "orange marks atoms/bonds changed, removed, or introduced."
                        if _normalize_report_language(report_language) == "en"
                        else "RDKit MCS 对比：绿色=初始和候选共有骨架；橙色=新增、删除或变化的原子/键。"
                    ),
                    "smiles": smiles,
                    "node_id": node.get("node_id", ""),
                    "score": node.get("score"),
                    "uct_value": node.get("uct_value"),
                    "total_reward": node.get("total_reward"),
                    "visit_count": node.get("visit_count"),
                    "unreachable": bool(node.get("unreachable")),
                    "parent_smiles_list": node.get("parent_smiles_list", []),
                    "action": node.get("action", ""),
                    "critic_rationale": node.get("critic_rationale", ""),
                    "generator_rationale": node.get("generator_rationale", ""),
                    "generator_confidence_score": node.get("generator_confidence_score", ""),
                    "agent_type": node.get("agent_type"),
                    "iteration_count": node.get("iteration_count"),
                    "properties": node.get("properties", {}) or {},
                    "rdkit_properties": props,
                    "image_base64": molecule_png_base64(
                        smiles,
                        size=(720, 560),
                        asset_name=f"{asset_id}_topology",
                    ),
                }
            )
        return {
            "parent": {
                "asset_id": "parent",
                "image_placeholder": "{{image:parent}}",
                "topology_image_placeholder": "{{image:parent}}",
                "smiles": initial_smiles,
                "iupac": _normalize_molecule_fact_value(initial_iupac, initial_smiles, fact_key="iupac"),
                "fragments": _normalize_molecule_fact_value(initial_fragments, initial_smiles, fact_key="fragments"),
                "rdkit_properties": parent_props,
                "image_base64": parent_image,
            },
            "candidates": candidate_assets,
        }

    def _report_assets_for_prompt(self, report_assets: Dict[str, Any]) -> Dict[str, Any]:
        """Return asset metadata without base64 payloads for LLM prompts."""
        result = {
            "parent": dict(report_assets.get("parent") or {}),
            "candidates": [],
        }
        result["parent"].pop("image_base64", None)
        for item in report_assets.get("candidates") or []:
            summary = dict(item)
            summary.pop("image_base64", None)
            summary.pop("topology_image_base64", None)
            summary.pop("diff_image_base64", None)
            result["candidates"].append(summary)
        return result

    async def _build_report_blueprint(
        self,
        *,
        state: MCGSState,
        candidate_nodes: Sequence[Dict[str, Any]],
        graph_statistics: Dict[str, Any],
        report_assets: Dict[str, Any],
        coverage_mode: str,
        top_n: int,
        first_round_n: int,
        max_detailed_cards: Optional[int],
        style_reference: str,
        report_language: str,
    ) -> ReportBlueprint:
        if self.report_agent is None:
            raise RuntimeError("ReportAgent is not available.")
        prompt_kwargs = {
            "user_prompt": state.get("user_prompt", ""),
            "optimization_goal": state.get("optimization_goal", ""),
            "project_manager_brief": state.get("project_manager_brief", ""),
            "initial_smiles": state.get("initial_smiles", ""),
            "initial_iupac": state.get("initial_iupac", ""),
            "initial_fragments": state.get("initial_fragments", ""),
            "graph_statistics": graph_statistics,
            "candidate_nodes": list(candidate_nodes),
            "report_assets": self._report_assets_for_prompt(report_assets),
            "current_info_list": state.get("current_info_list", []),
            "admet_policy": _extract_admet_policy(state),
            "coverage_mode": coverage_mode,
            "top_n_focus": top_n,
            "first_round_n": first_round_n,
            "max_detailed_cards": max_detailed_cards,
            "candidate_count": len(candidate_nodes),
            "style_reference": style_reference,
            "report_language": report_language,
            "report_validation_error": "",
        }
        last_error: Optional[Exception] = None
        for attempt in range(2):
            blueprint = await self.report_agent.plan_report(**prompt_kwargs)
            try:
                return self._validate_and_normalize_blueprint(
                    blueprint,
                    report_assets=report_assets,
                    candidate_nodes=candidate_nodes,
                    report_language=report_language,
                    max_detailed_cards=max_detailed_cards,
                    first_round_n=first_round_n,
                )
            except ValueError as exc:
                last_error = exc
                prompt_kwargs["report_validation_error"] = str(exc)
                if attempt == 0:
                    continue
                raise
        raise RuntimeError(f"ReportBlueprint validation failed: {last_error}")

    async def _build_agentic_html(
        self,
        *,
        state: MCGSState,
        blueprint: ReportBlueprint,
        report_assets: Dict[str, Any],
        candidate_nodes: Sequence[Dict[str, Any]],
        graph_statistics: Dict[str, Any],
        coverage_mode: str,
        top_n: int,
        max_detailed_cards: Optional[int],
        style_reference: str,
        report_language: str,
    ) -> str:
        if self.report_agent is None:
            raise RuntimeError("ReportAgent is not available.")
        task_context = {
            "user_prompt": state.get("user_prompt", ""),
            "optimization_goal": state.get("optimization_goal", ""),
            "project_manager_brief": state.get("project_manager_brief", ""),
            "graph_statistics": graph_statistics,
            "coverage_mode": coverage_mode,
            "top_n_focus": top_n,
            "first_round_n": len(blueprint.first_round_synthesis),
            "max_detailed_cards": max_detailed_cards,
            "style_reference": style_reference,
            "candidate_count": len(candidate_nodes),
            "report_language": report_language,
            "admet_policy": _extract_admet_policy(state),
        }
        return await self.report_agent.write_html_report(
            blueprint=blueprint,
            report_assets=self._report_assets_for_prompt(report_assets),
            task_context=task_context,
        )

    def _build_fallback_result(
        self,
        *,
        state: MCGSState,
        candidate_nodes: Sequence[Dict[str, Any]],
        initial_smiles: str,
        initial_iupac: Any,
        initial_fragments: Any,
        first_round_n: int,
        report_language: str = "zh",
    ) -> ReportResult:
        report = normalize_report_content(
            build_fallback_report(
                initial_smiles=initial_smiles,
                optimization_goal=state.get("optimization_goal", ""),
                candidate_nodes=candidate_nodes,
                report_language=report_language,
            ),
            initial_smiles=initial_smiles,
            optimization_goal=state.get("optimization_goal", ""),
            candidate_nodes=candidate_nodes,
            report_language=report_language,
        )
        report = self._limit_first_round(report, first_round_n)
        html_text = build_report_html(
            report,
            initial_smiles=initial_smiles,
            initial_iupac=initial_iupac,
            initial_fragments=initial_fragments,
            optimization_goal=state.get("optimization_goal", ""),
            candidate_nodes=candidate_nodes,
            report_language=report_language,
        )
        return ReportResult(html=html_text, output_path="", narrative=report)

    def _extract_candidate_nodes(self, mol_graph: Any) -> List[Dict[str, Any]]:
        if not mol_graph:
            return []

        root_node = getattr(mol_graph, "root_node", None)
        candidates = []
        for node in mol_graph.get_all_nodes():
            if root_node and getattr(node, "id", None) == getattr(root_node, "id", None):
                continue
            candidates.append(self._node_to_candidate(mol_graph, node))

        candidates.sort(
            key=lambda item: (
                bool(item.get("unreachable")),
                -float(item.get("score") or 0.0),
                -float(item.get("uct_value") or 0.0),
                int(item.get("iteration_count") or 0),
            )
        )
        return candidates

    def _node_to_candidate(self, mol_graph: Any, node: Any) -> Dict[str, Any]:
        parents = mol_graph.get_parents(node) if hasattr(mol_graph, "get_parents") else []
        parent = parents[0] if parents else None
        parent_id = getattr(parent, "id", None) if parent is not None else None
        parent_smiles = getattr(parent, "smiles", None) if parent is not None else None

        preferred_keys = [key for key in (parent_id, parent_smiles) if key]
        return {
            "node_id": getattr(node, "id", ""),
            "smiles": getattr(node, "smiles", ""),
            "score": getattr(node, "intrinsic_score", 0.0),
            "total_reward": getattr(node, "total_reward", 0.0),
            "visit_count": getattr(node, "visit_count", 0),
            "uct_value": getattr(node, "uct_value", 0.0),
            "unreachable": getattr(node, "unreachable", False),
            "parent_smiles_list": [getattr(item, "smiles", "") for item in parents],
            "action": self._first_parent_value(getattr(node, "actions_from_parents", {}), preferred_keys),
            "critic_rationale": self._first_parent_value(
                getattr(node, "rationale_from_parents_critic", {}),
                preferred_keys,
            ),
            "generator_rationale": self._first_parent_value(
                getattr(node, "rationale_from_parents_generator", {}),
                preferred_keys,
            ),
            "generator_confidence_score": self._first_parent_value(
                getattr(node, "confidence_score_from_parents_generator", {}),
                preferred_keys,
            ),
            "properties": getattr(node, "properties", {}) or {},
            "iteration_count": getattr(node, "iteration", 0),
            "selected_reason": getattr(node, "selected_reason", []),
            "agent_type": getattr(node, "agent_type", None),
        }

    def _extract_graph_statistics(self, mol_graph: Any) -> Dict[str, Any]:
        if not mol_graph:
            return {}
        if hasattr(mol_graph, "get_search_statistics"):
            try:
                return mol_graph.get_search_statistics()
            except Exception:
                pass
        nodes = mol_graph.get_all_nodes() if hasattr(mol_graph, "get_all_nodes") else []
        return {
            "total_nodes": len(nodes),
            "iteration_count": getattr(mol_graph, "iteration_count", None),
            "root_smiles": getattr(getattr(mol_graph, "root_node", None), "smiles", None),
        }

    def _extract_fallback_candidates(self, state: MCGSState) -> List[Dict[str, Any]]:
        raw_items = state.get("critic_evaluation", []) or []
        if isinstance(raw_items, str):
            try:
                raw_items = json.loads(raw_items)
            except json.JSONDecodeError:
                return []
        if isinstance(raw_items, dict):
            raw_items = raw_items.get("optimizations", [])

        candidates = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            smiles = item.get("smiles") or item.get("SMILES")
            if not smiles:
                continue
            candidates.append(
                {
                    "node_id": item.get("node_id", ""),
                    "smiles": smiles,
                    "score": item.get("score", 0.0),
                    "total_reward": item.get("total_reward", item.get("score", 0.0)),
                    "visit_count": item.get("visit_count", 0),
                    "uct_value": item.get("uct_value", 0.0),
                    "unreachable": item.get("unreachable", False),
                    "parent_smiles_list": item.get("parent_smiles_list", []),
                    "action": item.get("action") or item.get("modification_type", ""),
                    "critic_rationale": item.get("critic_rationale") or item.get("rationale", ""),
                    "generator_rationale": item.get("generator_rationale", ""),
                    "generator_confidence_score": item.get("generator_confidence_score", ""),
                    "properties": item.get("properties", {}) or {},
                    "iteration_count": item.get("iteration_count", item.get("iteration", 0)),
                    "selected_reason": item.get("selected_reason", []),
                    "agent_type": item.get("agent_type"),
                }
            )
        candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return candidates

    def _dedupe_candidates(self, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        result = []
        for item in candidates:
            smiles = item.get("smiles") or item.get("SMILES")
            if not smiles or smiles in seen:
                continue
            seen.add(smiles)
            normalized = dict(item)
            normalized["smiles"] = smiles
            result.append(normalized)
        return result

    def _first_parent_value(self, values: Dict[str, Any], preferred_keys: Sequence[str]) -> Any:
        if not values:
            return ""
        for key in preferred_keys:
            if key in values:
                return values[key]
        return next(iter(values.values()))

    def _limit_first_round(self, report: OptimizationReport, first_round_n: int) -> OptimizationReport:
        if first_round_n <= 0:
            return report
        report_dict = report.model_dump()
        report_dict["first_round_synthesis"] = report_dict.get("first_round_synthesis", [])[:first_round_n]
        return OptimizationReport.model_validate(report_dict)

    def _limit_blueprint_first_round(self, blueprint: ReportBlueprint, first_round_n: int) -> ReportBlueprint:
        if first_round_n <= 0:
            return blueprint
        blueprint_dict = blueprint.model_dump()
        blueprint_dict["first_round_synthesis"] = blueprint_dict.get("first_round_synthesis", [])[:first_round_n]
        return ReportBlueprint.model_validate(blueprint_dict)

    def _validate_and_normalize_blueprint(
        self,
        blueprint: ReportBlueprint,
        *,
        report_assets: Dict[str, Any],
        candidate_nodes: Sequence[Dict[str, Any]],
        report_language: str,
        max_detailed_cards: Optional[int],
        first_round_n: int,
    ) -> ReportBlueprint:
        """Keep the model-authored blueprint locked to the current MCGS graph."""
        language = _normalize_report_language(report_language)
        candidate_smiles = [str(node.get("smiles") or "").strip() for node in candidate_nodes if node.get("smiles")]
        candidate_smiles_set = set(candidate_smiles)
        node_by_smiles = {str(node.get("smiles") or "").strip(): node for node in candidate_nodes if node.get("smiles")}
        asset_by_smiles = {
            str(item.get("smiles") or "").strip(): item
            for item in report_assets.get("candidates") or []
            if item.get("smiles")
        }
        asset_ids = {str(item.get("asset_id") or "").strip() for item in report_assets.get("candidates") or []}

        if not candidate_smiles:
            raise ValueError("ReportBlueprint cannot be validated because the current MCGS graph has no candidates.")

        blueprint_dict = blueprint.model_dump()
        blueprint_dict["report_language"] = language

        unknown_complete = [
            str(smiles or "").strip()
            for smiles in blueprint_dict.get("complete_candidate_smiles") or []
            if str(smiles or "").strip() and str(smiles or "").strip() not in candidate_smiles_set
        ]
        if unknown_complete:
            raise ValueError(
                "ReportBlueprint complete_candidate_smiles contains graph-external SMILES: "
                + ", ".join(unknown_complete[:5])
            )
        blueprint_dict["complete_candidate_smiles"] = candidate_smiles

        groups = _normalize_blueprint_groups(blueprint_dict.get("strategy_groups") or [], language)
        group_ids = [group["group_id"] for group in groups if group.get("group_id")]
        if not group_ids:
            groups = _fallback_blueprint_groups(language)
            group_ids = [group["group_id"] for group in groups]

        used_design_ids: set[str] = set()
        seen_smiles: set[str] = set()
        cards: List[Dict[str, Any]] = []
        for raw_card in blueprint_dict.get("molecule_cards") or []:
            card = raw_card.model_dump() if hasattr(raw_card, "model_dump") else dict(raw_card)
            smiles = str(card.get("smiles") or "").strip()
            if not smiles:
                continue
            if smiles not in candidate_smiles_set:
                raise ValueError(f"ReportBlueprint molecule_cards contains graph-external SMILES: {smiles}")
            asset = asset_by_smiles.get(smiles)
            if not asset:
                raise ValueError(f"ReportBlueprint molecule card has no matching report asset for SMILES: {smiles}")
            if str(card.get("asset_id") or "").strip() not in asset_ids:
                card["asset_id"] = asset.get("asset_id")
            card["asset_id"] = asset.get("asset_id")
            if smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)
            node = node_by_smiles.get(smiles, {})
            group_id = _resolve_group_id(str(card.get("strategy_group") or ""), group_ids)
            if not group_id:
                group_id = _fallback_group_for_candidate(node, candidate_smiles.index(smiles) + 1)
                if group_id not in group_ids:
                    groups.append(_fallback_blueprint_group(group_id, language))
                    group_ids.append(group_id)
            card["strategy_group"] = group_id
            design_id = str(card.get("design_id") or "").strip()
            if not design_id or design_id in used_design_ids:
                design_id = _next_blueprint_design_id(group_id, used_design_ids)
            card["design_id"] = design_id
            used_design_ids.add(design_id)
            if bool(node.get("unreachable")) and card.get("priority") == "H":
                card["priority"] = "M"
            card.setdefault("role_tags", [])
            cards.append(card)

        target_cards = _resolve_max_detailed_cards(
            len(candidate_smiles),
            max_detailed_cards=max_detailed_cards,
            top_n=0,
            first_round_n=first_round_n,
        )
        for group_id in list(group_ids):
            if any(_resolve_group_id(str(card.get("strategy_group") or ""), group_ids) == group_id for card in cards):
                continue
            node = next(
                (
                    item
                    for item in candidate_nodes
                    if str(item.get("smiles") or "").strip() not in seen_smiles
                    and not bool(item.get("unreachable"))
                ),
                None,
            )
            if node is None:
                node = next(
                    (
                        item
                        for item in candidate_nodes
                        if str(item.get("smiles") or "").strip() not in seen_smiles
                    ),
                    None,
                )
            if node is None:
                continue
            card = _fallback_blueprint_card(
                node,
                asset_by_smiles=asset_by_smiles,
                design_id=_next_blueprint_design_id(group_id, used_design_ids),
                group_id=group_id,
                language=language,
            )
            used_design_ids.add(card["design_id"])
            seen_smiles.add(card["smiles"])
            cards.append(card)

        for index, node in enumerate(candidate_nodes, start=1):
            if len(cards) >= target_cards:
                break
            smiles = str(node.get("smiles") or "").strip()
            if not smiles or smiles in seen_smiles:
                continue
            group_id = _fallback_group_for_candidate(node, index)
            if group_id not in group_ids:
                groups.append(_fallback_blueprint_group(group_id, language))
                group_ids.append(group_id)
            card = _fallback_blueprint_card(
                node,
                asset_by_smiles=asset_by_smiles,
                design_id=_next_blueprint_design_id(group_id, used_design_ids),
                group_id=group_id,
                language=language,
            )
            used_design_ids.add(card["design_id"])
            seen_smiles.add(smiles)
            cards.append(card)

        blueprint_dict["strategy_groups"] = groups
        blueprint_dict["molecule_cards"] = cards
        blueprint_dict["first_round_synthesis"] = _normalize_blueprint_first_round(
            blueprint_dict.get("first_round_synthesis") or [],
            cards,
            node_by_smiles=node_by_smiles,
            first_round_n=first_round_n,
            language=language,
        )
        if not blueprint_dict.get("risk_notes"):
            blueprint_dict["risk_notes"] = _fallback_blueprint_risk_notes(language)
        if not blueprint_dict.get("expert_questions"):
            blueprint_dict["expert_questions"] = _fallback_blueprint_expert_questions(language)
        return ReportBlueprint.model_validate(blueprint_dict)

    def _default_filename(self, state: MCGSState, mol_graph: Any) -> str:
        run_timestamp = getattr(mol_graph, "run_timestamp", None) if mol_graph else None
        timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_filename_part(state.get('initial_smiles', ''))}_{timestamp}_SAR_design_report.html"

    def _write_html(self, *, html_text: str, output_dir: Path, filename: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        output_path.write_text(html_text, encoding="utf-8")
        return output_path


def safe_filename_part(value: str, max_length: int = 96) -> str:
    """Convert a SMILES/title fragment into a filesystem-safe filename part."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "molecule").strip("_")
    if not cleaned:
        cleaned = "molecule"
    if len(cleaned) <= max_length:
        return cleaned
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:max_length]}_{digest}"


def calculate_molecule_properties(smiles: str) -> Dict[str, Any]:
    """Calculate report table properties with RDKit."""
    mol, _canonical, error = parse_smiles_quiet(smiles)
    if mol is None or error:
        return {
            "valid": False,
            "invalid_reason": error or "RDKit could not parse SMILES",
            "MW": None,
            "cLogP": None,
            "TPSA": None,
            "HBD": None,
            "HBA": None,
            "ArR": None,
            "RotB": None,
            "Fsp3": None,
            "Ro5": False,
        }

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    ro5 = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
    return {
        "valid": True,
        "MW": round(mw, 1),
        "cLogP": round(logp, 2),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 1),
        "HBD": hbd,
        "HBA": hba,
        "ArR": rdMolDescriptors.CalcNumAromaticRings(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "Fsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 2),
        "Ro5": ro5,
    }


def molecule_png_base64(
    smiles: str,
    size: Tuple[int, int] = (320, 260),
    *,
    asset_name: Optional[str] = None,
) -> str:
    """Return a base64 PNG structure depiction for a SMILES string."""
    mol, _canonical, error = parse_smiles_quiet(smiles)
    if mol is None or error:
        return ""
    image = Draw.MolToImage(mol, size=size)
    return _png_image_to_base64(
        image,
        asset_name=asset_name,
        content_key=f"topology|{smiles}|{size[0]}x{size[1]}",
    )


def molecule_comparison_png_base64(
    parent_smiles: str,
    candidate_smiles: str,
    size: Tuple[int, int] = (960, 420),
    *,
    mcs_timeout: int = REPORT_MCS_TIMEOUT_SECONDS,
    asset_name: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return an RDKit before/after PNG with MCS and changed regions highlighted."""
    parent_mol, _parent_canonical, parent_error = parse_smiles_quiet(parent_smiles)
    candidate_mol, _candidate_canonical, candidate_error = parse_smiles_quiet(candidate_smiles)
    if parent_mol is None or parent_error or candidate_mol is None or candidate_error:
        return "", {
            "valid": False,
            "invalid_reason": parent_error or candidate_error or "RDKit could not parse one or both SMILES",
        }

    parent_mol = Chem.Mol(parent_mol)
    candidate_mol = Chem.Mol(candidate_mol)
    summary: Dict[str, Any] = {"valid": True, "method": "mcs"}

    try:
        mcs = rdFMCS.FindMCS(
            [parent_mol, candidate_mol],
            timeout=max(1, int(mcs_timeout or REPORT_MCS_TIMEOUT_SECONDS)),
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            matchValences=True,
        )
    except Exception as exc:
        mcs = None
        summary.update({"method": "plain", "mcs_error": f"{type(exc).__name__}: {exc}"})

    parent_match: Tuple[int, ...] = ()
    candidate_match: Tuple[int, ...] = ()
    if mcs is not None and getattr(mcs, "smartsString", ""):
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is not None:
            parent_match = tuple(parent_mol.GetSubstructMatch(mcs_mol))
            candidate_match = tuple(candidate_mol.GetSubstructMatch(mcs_mol))

    if parent_match and candidate_match:
        _compute_aligned_2d_coords(parent_mol, candidate_mol, parent_match, candidate_match)
        parent_atoms, parent_bonds, parent_atom_colors, parent_bond_colors, parent_changed_atoms = _highlight_mcs_difference(
            parent_mol,
            parent_match,
        )
        candidate_atoms, candidate_bonds, candidate_atom_colors, candidate_bond_colors, candidate_changed_atoms = _highlight_mcs_difference(
            candidate_mol,
            candidate_match,
        )
        summary.update(
            {
                "method": "mcs",
                "mcs_atoms": len(candidate_match),
                "mcs_bonds": getattr(mcs, "numBonds", 0),
                "parent_changed_atoms": len(parent_changed_atoms),
                "candidate_changed_atoms": len(candidate_changed_atoms),
                "mcs_canceled": bool(getattr(mcs, "canceled", False)),
            }
        )
        image = Draw.MolsToGridImage(
            [parent_mol, candidate_mol],
            molsPerRow=2,
            subImgSize=(max(240, size[0] // 2), size[1]),
            legends=["Initial: green common / orange removed", "Candidate: orange changed/new"],
            highlightAtomLists=[parent_atoms, candidate_atoms],
            highlightBondLists=[parent_bonds, candidate_bonds],
            highlightAtomColors=[parent_atom_colors, candidate_atom_colors],
            highlightBondColors=[parent_bond_colors, candidate_bond_colors],
            useSVG=False,
        )
    else:
        for mol in (parent_mol, candidate_mol):
            AllChem.Compute2DCoords(mol)
        summary.update({"method": "plain", "mcs_atoms": 0, "mcs_bonds": 0})
        image = Draw.MolsToGridImage(
            [parent_mol, candidate_mol],
            molsPerRow=2,
            subImgSize=(max(240, size[0] // 2), size[1]),
            legends=["Initial", "Candidate"],
            useSVG=False,
        )

    image_base64 = _png_image_to_base64(
        image,
        asset_name=asset_name,
        content_key=f"diff|{parent_smiles}|{candidate_smiles}|{size[0]}x{size[1]}|{summary.get('method')}",
    )
    return image_base64, summary


def _png_image_to_base64(image: Any, *, asset_name: Optional[str], content_key: str) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    if asset_name:
        _write_report_intermediate_png(asset_name, content_key, png_bytes)
    return base64.b64encode(png_bytes).decode("ascii")


def _write_report_intermediate_png(asset_name: str, content_key: str, png_bytes: bytes) -> Path:
    output_dir = _current_report_tmp_dir() / REPORT_ASSET_IMAGE_TMP_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1((content_key or asset_name).encode("utf-8")).hexdigest()[:12]
    filename = f"{safe_filename_part(asset_name, max_length=80)}_{digest}.png"
    output_path = output_dir / filename
    output_path.write_bytes(png_bytes)
    return output_path


def _current_report_tmp_dir() -> Path:
    value = os.getenv("M3OS_TMP_DIR")
    if not value or not value.strip():
        return PROJECT_TMP_DIR
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _compute_aligned_2d_coords(
    parent_mol: Chem.Mol,
    candidate_mol: Chem.Mol,
    parent_match: Sequence[int],
    candidate_match: Sequence[int],
) -> None:
    AllChem.Compute2DCoords(parent_mol)
    try:
        AllChem.GenerateDepictionMatching2DStructure(
            candidate_mol,
            parent_mol,
            atomMap=list(zip(candidate_match, parent_match)),
        )
    except Exception:
        AllChem.Compute2DCoords(candidate_mol)


def _highlight_mcs_difference(
    mol: Chem.Mol,
    common_atoms: Sequence[int],
) -> Tuple[
    List[int],
    List[int],
    Dict[int, Tuple[float, float, float]],
    Dict[int, Tuple[float, float, float]],
    List[int],
]:
    common_atom_set = set(common_atoms)
    changed_atom_set = {atom.GetIdx() for atom in mol.GetAtoms()} - common_atom_set
    common_bond_set = set()
    changed_bond_set = set()
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in common_atom_set and end in common_atom_set:
            common_bond_set.add(bond.GetIdx())
        elif begin in changed_atom_set or end in changed_atom_set:
            changed_bond_set.add(bond.GetIdx())

    highlighted_atoms = common_atom_set | changed_atom_set
    highlighted_bonds = common_bond_set | changed_bond_set
    atom_colors = {
        atom_idx: (CHANGE_HIGHLIGHT_COLOR if atom_idx in changed_atom_set else COMMON_HIGHLIGHT_COLOR)
        for atom_idx in highlighted_atoms
    }
    bond_colors = {
        bond_idx: (CHANGE_HIGHLIGHT_COLOR if bond_idx in changed_bond_set else COMMON_HIGHLIGHT_COLOR)
        for bond_idx in highlighted_bonds
    }
    return (
        sorted(highlighted_atoms),
        sorted(highlighted_bonds),
        atom_colors,
        bond_colors,
        sorted(changed_atom_set),
    )


def prepare_agentic_report_html(
    html_text: str,
    *,
    report_assets: Dict[str, Any],
    candidate_nodes: Sequence[Dict[str, Any]],
    report_language: str = "en",
) -> str:
    """Validate, sanitize, and complete agent-authored report HTML."""
    language = _normalize_report_language(report_language)
    cleaned = _strip_html_fences(html_text)
    has_diff_placeholder = bool(re.search(r"\{\{image_diff:[^}]+\}\}", cleaned))
    expanded = _expand_image_placeholders(cleaned, report_assets)
    soup = BeautifulSoup(expanded, "html.parser")
    _ensure_required_html_parts(soup, report_language=language)
    _ensure_report_visual_css(soup)
    _sanitize_report_soup(soup)
    _assert_no_unknown_report_smiles(soup, report_assets=report_assets, candidate_nodes=candidate_nodes)
    if not has_diff_placeholder:
        _append_structure_diff_gallery(soup, report_assets, report_language=language)
    if _missing_candidate_smiles(soup, candidate_nodes) or _missing_complete_candidate_overview(soup):
        _append_complete_candidate_overview(
            soup,
            report_assets,
            candidate_nodes,
            report_language=language,
        )
    rendered = re.sub(r"(?is)^\s*<!doctype[^>]*>\s*", "", str(soup).lstrip())
    return "<!doctype html>\n" + rendered


def _strip_html_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("html"):
            cleaned = cleaned[4:].strip()
    return cleaned


def _expand_image_placeholders(html_text: str, report_assets: Dict[str, Any]) -> str:
    image_map: Dict[str, str] = {}
    parent = report_assets.get("parent") or {}
    if parent.get("image_base64"):
        image_map["parent"] = f"data:image/png;base64,{parent['image_base64']}"
    for item in report_assets.get("candidates") or []:
        asset_id = str(item.get("asset_id") or "").strip()
        image = item.get("image_base64")
        if asset_id and image:
            image_map[asset_id] = f"data:image/png;base64,{image}"

    def _replace_image(match: re.Match[str]) -> str:
        asset_id = match.group(1).strip()
        return image_map.get(asset_id, "")

    def _replace_diff_image(match: re.Match[str]) -> str:
        asset_id = match.group(1).strip()
        return _diff_image_data_uri(report_assets, asset_id)

    expanded = re.sub(r"\{\{image_diff:([^}]+)\}\}", _replace_diff_image, html_text)
    return re.sub(r"\{\{image:([^}]+)\}\}", _replace_image, expanded)


def _diff_image_data_uri(report_assets: Dict[str, Any], asset_id: str) -> str:
    parent = report_assets.get("parent") or {}
    parent_smiles = str(parent.get("smiles") or "").strip()
    if not parent_smiles:
        return ""

    for item in report_assets.get("candidates") or []:
        if str(item.get("asset_id") or "").strip() != asset_id:
            continue
        if "diff_image_base64" not in item:
            image_base64, summary = molecule_comparison_png_base64(
                parent_smiles,
                str(item.get("smiles") or "").strip(),
                asset_name=f"{asset_id}_diff",
            )
            item["diff_image_base64"] = image_base64
            item["diff_summary"] = summary
        image_base64 = str(item.get("diff_image_base64") or "")
        return f"data:image/png;base64,{image_base64}" if image_base64 else ""
    return ""


def _ensure_report_visual_css(soup: BeautifulSoup) -> None:
    head = soup.find("head")
    if head is None:
        return

    style = head.find("style")
    if style is None:
        style = soup.new_tag("style")
        head.append(style)

    current_css = style.string if style.string is not None else style.get_text()
    if "structure-diff-gallery" in current_css and "legend-dot" in current_css:
        return
    style.string = f"{current_css or ''}\n{_report_visual_css()}".strip()


def _append_structure_diff_gallery(
    soup: BeautifulSoup,
    report_assets: Dict[str, Any],
    *,
    report_language: str = "zh",
) -> None:
    body = soup.find("body")
    if body is None:
        raise ValueError("Reporter HTML must contain body before appending structure comparison gallery.")
    if body.find(id="structure-diff-gallery") is not None:
        return

    cards = []
    for item in (report_assets.get("candidates") or [])[:MAX_AUTO_DIFF_GALLERY_IMAGES]:
        asset_id = str(item.get("asset_id") or "").strip()
        smiles = str(item.get("smiles") or "").strip()
        if not asset_id or not smiles:
            continue
        image_src = _diff_image_data_uri(report_assets, asset_id)
        if not image_src:
            continue
        score_text = _fmt_number(item.get("score")) if item.get("score") is not None else ""
        score_badge = f"<span>Score {_e(score_text)}</span>" if score_text else ""
        action = _compact_text(item.get("action", ""), default="")
        action_label = "Action" if _normalize_report_language(report_language) == "en" else "Action"
        action_line = f'<p class="diff-summary"><b>{action_label}</b>: {_e(action)}</p>' if action else ""
        alt_text = (
            f"{asset_id} initial and candidate topology comparison"
            if _normalize_report_language(report_language) == "en"
            else f"{asset_id} 初始分子与候选分子二维拓扑对比"
        )
        cards.append(
            "\n".join(
                [
                    '<article class="structure-diff-card">',
                    f"<header><strong>{_e(asset_id)}</strong>{score_badge}</header>",
                    f'<img src="{image_src}" alt="{_e(alt_text)}"/>',
                    f'<div class="mono">{_e(smiles)}</div>',
                    f'<p class="diff-summary">{_e(_format_diff_summary(item.get("diff_summary") or {}, report_language=report_language))}</p>',
                    action_line,
                    "</article>",
                ]
            )
        )

    if not cards:
        return

    fragment = BeautifulSoup(
        "\n".join(
            [
                '<section id="structure-diff-gallery" class="structure-diff-gallery">',
                (
                    "<h2>Structure Change Highlights</h2>"
                    if _normalize_report_language(report_language) == "en"
                    else "<h2>结构变化高亮</h2>"
                ),
                _visual_legend_html(report_language=report_language),
                '<div class="structure-diff-grid">',
                *cards,
                "</div>",
                "</section>",
            ]
        ),
        "html.parser",
    )
    body.append(fragment)


def _visual_legend_html(*, report_language: str = "zh") -> str:
    if _normalize_report_language(report_language) == "en":
        return (
            '<div class="visual-legend">'
            '<span><i class="legend-dot common"></i>Green: shared scaffold</span>'
            '<span><i class="legend-dot change"></i>Orange: changed region versus root</span>'
            "</div>"
        )
    return (
        '<div class="visual-legend">'
        '<span><i class="legend-dot common"></i>绿色：公共骨架</span>'
        '<span><i class="legend-dot change"></i>橙色：相对初始分子的变化</span>'
        "</div>"
    )


def _format_diff_summary(summary: Dict[str, Any], *, report_language: str = "zh") -> str:
    if _normalize_report_language(report_language) == "en":
        if not summary.get("valid", False):
            return "RDKit could not compute a structure-difference depiction for this candidate."
        if summary.get("method") != "mcs":
            return "No stable MCS was found, so the unhighlighted initial/candidate topologies are shown."
        return (
            f"MCS {summary.get('mcs_atoms', 0)} atoms / {summary.get('mcs_bonds', 0)} bonds; "
            f"initial changed atoms {summary.get('parent_changed_atoms', 0)}, "
            f"candidate changed atoms {summary.get('candidate_changed_atoms', 0)}."
        )
    if not summary.get("valid", False):
        return "RDKit 未能计算该候选与初始分子的结构差异图。"
    if summary.get("method") != "mcs":
        return "未找到稳定 MCS，展示未高亮的初始/候选二维拓扑。"
    return (
        f"MCS {summary.get('mcs_atoms', 0)} atoms / {summary.get('mcs_bonds', 0)} bonds；"
        f"初始变化原子 {summary.get('parent_changed_atoms', 0)}，"
        f"候选变化原子 {summary.get('candidate_changed_atoms', 0)}。"
    )


def _report_visual_css() -> str:
    return (
        ".structure-diff-gallery{margin-top:34px}"
        ".structure-diff-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}"
        ".structure-diff-card{background:#101820;border:1px solid #284052;border-radius:8px;padding:12px;overflow:hidden;box-shadow:0 18px 45px rgba(0,0,0,.22)}"
        ".structure-diff-card header{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:8px;font-size:12px;color:#b9c7d6}"
        ".structure-diff-card img,.diffbox img,.mol-visuals img{display:block;max-width:100%;height:auto;margin:0 auto;border:1px solid #314655;border-radius:6px;background:#fff}"
        ".structure-diff-card .mono,.diffbox .mono{font-family:Consolas,monospace;font-size:11px;color:#d6e1ea;word-break:break-all;margin-top:8px}"
        ".visual-legend{font-size:12px;color:#aab8c5;display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0 12px}"
        ".visual-legend span{display:inline-flex;align-items:center;gap:4px}"
        ".legend-dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle}"
        ".legend-dot.common{background:#61b36b}.legend-dot.change{background:#ed7a2e}"
        ".diff-summary{font-size:11.5px;color:#aebdca;margin:7px 0 0;line-height:1.42}"
        ".mol-visuals{display:grid;grid-template-columns:1fr;gap:8px;margin:8px 0}"
        ".visual-title{font-weight:700;font-size:12px;color:#c9d8e6;margin:4px 0}"
        ".topologybox,.diffbox{border:1px solid #293c4c;border-radius:8px;background:#0e1720;padding:8px;margin:8px 0}"
    )


def _ensure_required_html_parts(soup: BeautifulSoup, *, report_language: str = "zh") -> None:
    html_tag = soup.find("html")
    body = soup.find("body")
    if html_tag is None or body is None:
        raise ValueError("Reporter HTML must contain html and body tags.")
    html_tag["lang"] = _normalize_report_language(report_language)

    head = soup.find("head")
    if head is None:
        head = soup.new_tag("head")
        html_tag.insert(0, head)

    meta_charset = head.find("meta", attrs={"charset": True})
    if meta_charset is None:
        meta_charset = soup.new_tag("meta")
        meta_charset["charset"] = "utf-8"
        head.insert(0, meta_charset)

    title = head.find("title")
    if title is None or not title.get_text(strip=True):
        if title is None:
            title = soup.new_tag("title")
            head.append(title)
        title.string = "SAR Optimization Design Report" if _normalize_report_language(report_language) == "en" else "SAR 优化设计报告"

    if head.find("style") is None:
        style = soup.new_tag("style")
        style.string = _agentic_default_css()
        head.append(style)


def _sanitize_report_soup(soup: BeautifulSoup) -> None:
    for tag_name in ("script", "iframe", "form", "object", "embed", "link"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for meta in soup.find_all("meta"):
        if str(meta.get("http-equiv") or "").lower() == "refresh":
            meta.decompose()

    for style in soup.find_all("style"):
        css = style.string if style.string is not None else style.get_text()
        css = re.sub(r"(?is)@import\s+[^;]+;?", "", css or "")
        css = re.sub(r"(?is)url\(\s*['\"]?(?:https?:|//|file:|javascript:)[^)]+\)", "none", css)
        style.string = css

    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if str(attr).lower().startswith("on"):
                del tag.attrs[attr]

        for attr in ("href", "src"):
            value = tag.get(attr)
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if _is_external_or_unsafe_url(stripped):
                if tag.name == "img":
                    tag.decompose()
                    break
                del tag.attrs[attr]
                continue
            if tag.name == "img" and stripped and not stripped.startswith("data:image/png;base64,"):
                tag.decompose()
                break


def _is_external_or_unsafe_url(value: str) -> bool:
    lowered = value.lower()
    return (
        lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("//")
        or lowered.startswith("javascript:")
        or lowered.startswith("file:")
        or lowered.startswith("data:text/html")
    )


def _missing_candidate_smiles(soup: BeautifulSoup, candidate_nodes: Sequence[Dict[str, Any]]) -> List[str]:
    rendered = html.unescape(soup.get_text("\n") + "\n" + str(soup))
    missing = []
    for node in candidate_nodes:
        smiles = str(node.get("smiles") or "").strip()
        if smiles and smiles not in rendered:
            missing.append(smiles)
    return missing


def _missing_complete_candidate_overview(soup: BeautifulSoup) -> bool:
    if soup.find(id="complete-candidate-overview") is not None:
        return False
    if soup.select_one("table.complete-candidates") is not None:
        return False
    return True


def _assert_no_unknown_report_smiles(
    soup: BeautifulSoup,
    *,
    report_assets: Dict[str, Any],
    candidate_nodes: Sequence[Dict[str, Any]],
) -> None:
    """Reject parseable SMILES tokens in molecule/code fields when they are not in the graph."""
    allowed_exact = {
        str((report_assets.get("parent") or {}).get("smiles") or "").strip(),
        *[str(node.get("smiles") or "").strip() for node in candidate_nodes],
    }
    allowed_exact = {item for item in allowed_exact if item}
    allowed_canonical = {
        _canonical_report_smiles_key(item)
        for item in allowed_exact
        if _canonical_report_smiles_key(item)
    }
    suspicious: set[str] = set()
    for tag in soup.find_all(True):
        classes = tag.get("class") or []
        class_text = " ".join(str(item).lower() for item in classes)
        if tag.name != "code" and "mono" not in class_text and "smi" not in class_text and "smiles" not in class_text:
            continue
        text = html.unescape(tag.get_text(" ", strip=True))
        for match in REPORT_SMILES_TOKEN_PATTERN.finditer(text):
            token = match.group(1).strip().rstrip(".,;:")
            if token in allowed_exact:
                continue
            mol, canonical, error = parse_smiles_quiet(token)
            if mol is None or error:
                continue
            canonical_key = canonical or _canonical_report_smiles_key(token)
            if canonical_key and canonical_key in allowed_canonical:
                continue
            suspicious.add(token)
    if suspicious:
        raise ValueError(
            "Reporter HTML contains parseable graph-external SMILES in molecule/code fields: "
            + ", ".join(sorted(suspicious)[:8])
        )


def _append_complete_candidate_overview(
    soup: BeautifulSoup,
    report_assets: Dict[str, Any],
    candidate_nodes: Sequence[Dict[str, Any]],
    *,
    report_language: str = "zh",
) -> None:
    body = soup.find("body")
    if body is None:
        raise ValueError("Reporter HTML must contain body before appending overview.")
    existing_section = body.find(id="complete-candidate-overview")
    if existing_section is not None:
        existing_section.decompose()
    for existing_table in body.select("table.complete-candidates"):
        existing_table.decompose()

    asset_by_smiles = {
        str(item.get("smiles") or ""): item
        for item in report_assets.get("candidates") or []
    }
    rows = []
    for index, node in enumerate(candidate_nodes, start=1):
        smiles = str(node.get("smiles") or "").strip()
        asset = asset_by_smiles.get(smiles, {})
        props = asset.get("rdkit_properties") or {}
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{_e(asset.get('asset_id') or '')}</td>"
            f"<td class=\"mono\">{_e(smiles)}</td>"
            f"<td>{_e(_fmt_number(node.get('score')))}</td>"
            f"<td>{_e(_fmt_number(node.get('uct_value')))}</td>"
            f"<td>{_e(_format_property(props, 'MW'))}</td>"
            f"<td>{_e(_format_property(props, 'cLogP'))}</td>"
            f"<td>{_e(_format_property(props, 'TPSA'))}</td>"
            f"<td>{_e(_compact_text(node.get('action', ''), default=''))}</td>"
            "</tr>"
        )
    is_en = _normalize_report_language(report_language) == "en"
    fragment = BeautifulSoup(
        "\n".join(
            [
                '<section id="complete-candidate-overview">',
                "<h2>Complete Candidate Audit</h2>" if is_en else "<h2>完整候选总览</h2>",
                (
                    '<p class="legend">Automatically completed by the report generator to ensure every MCGS candidate SMILES is present.</p>'
                    if is_en
                    else '<p class="legend">该表由系统自动补充，确保 MCGS 图中所有候选 SMILES 均进入最终报告。</p>'
                ),
                '<table class="summary complete-candidates">',
                (
                    "<tr><th>#</th><th>Asset</th><th>SMILES</th><th>Score</th><th>UCT</th>"
                    "<th>MW</th><th>cLogP</th><th>TPSA</th><th>Action</th></tr>"
                ),
                *rows,
                "</table>",
                "</section>",
            ]
        ),
        "html.parser",
    )
    body.append(fragment)


def _agentic_default_css() -> str:
    return (
        'body{font-family:-apple-system,Helvetica,Arial,"Microsoft YaHei",sans-serif;'
        "margin:0;color:#d8e2ec;background:#081017;max-width:1720px;padding:28px}"
        "h1{margin:0 0 10px;color:#f3f7fb;letter-spacing:0}h2{border-left:4px solid #51c7a9;padding-left:12px;margin-top:34px;color:#eef6fb}"
        "section{margin-top:24px}.summary{border-collapse:collapse;width:100%;margin-top:10px;background:#0e1720}"
        ".summary th,.summary td{border:1px solid #263847;padding:7px;font-size:13px;vertical-align:top}"
        ".summary th{background:#172433;color:#dfeaf4}.mono{font-family:Consolas,monospace;word-break:break-all;color:#e6edf3}"
        ".legend{font-size:12px;color:#9fb0bf;margin-top:8px}"
        "a{color:#76d7c4}.card,.panel,article{border-radius:8px}"
    )


def normalize_report_content(
    report_content: Optional[Any],
    *,
    initial_smiles: str,
    optimization_goal: str,
    candidate_nodes: Sequence[Dict[str, Any]],
    report_language: str = "en",
) -> OptimizationReport:
    """Validate report content and ensure every graph candidate has a card."""
    report = _coerce_report(report_content)
    if report is None:
        report = build_fallback_report(
            initial_smiles=initial_smiles,
            optimization_goal=optimization_goal,
            candidate_nodes=candidate_nodes,
            report_language=report_language,
        )

    report_dict = report.model_dump()
    groups = list(report_dict.get("strategy_groups") or [])
    designs = list(report_dict.get("designed_molecules") or [])

    if not groups:
        groups = _fallback_strategy_groups(report_language)

    known_smiles = {item.get("smiles") for item in designs if item.get("smiles")}
    fallback_group_id = "Z"
    if any(node.get("smiles") not in known_smiles for node in candidate_nodes):
        if not any(group.get("group_id") == fallback_group_id for group in groups):
            is_en = _normalize_report_language(report_language) == "en"
            groups.append(
                {
                    "group_id": fallback_group_id,
                    "group_name": "Search-result supplement" if is_en else "搜索结果补充",
                    "design_purpose": (
                        "Retain MCGS candidates not grouped by the report agent."
                        if is_en else "保留 MCGS 生成但未被报告 agent 分组的候选分子。"
                    ),
                    "sar_output": (
                        "Cover every designed molecule in the final graph without omissions."
                        if is_en else "补充覆盖最终图中的全部设计分子，避免遗漏。"
                    ),
                }
            )

    next_index = 1
    used_ids = {item.get("design_id") for item in designs if item.get("design_id")}
    for node in candidate_nodes:
        smiles = node.get("smiles")
        if not smiles or smiles in known_smiles:
            continue
        design_id = f"{fallback_group_id}{next_index}"
        while design_id in used_ids:
            next_index += 1
            design_id = f"{fallback_group_id}{next_index}"
        next_index += 1
        used_ids.add(design_id)
        known_smiles.add(smiles)
        designs.append(_candidate_to_design(node, design_id, fallback_group_id, report_language))

    if not report_dict.get("recommended_modification_sites"):
        report_dict["recommended_modification_sites"] = _fallback_modification_sites(candidate_nodes, report_language)

    if not report_dict.get("risk_notes"):
        report_dict["risk_notes"] = _fallback_risk_notes(optimization_goal, report_language)

    first_round = list(report_dict.get("first_round_synthesis") or [])
    if not first_round:
        first_round = _fallback_first_round(designs, candidate_nodes, report_language=report_language)

    report_dict["strategy_groups"] = groups
    report_dict["designed_molecules"] = designs
    report_dict["first_round_synthesis"] = first_round
    return OptimizationReport.model_validate(report_dict)


def build_fallback_report(
    *,
    initial_smiles: str,
    optimization_goal: str,
    candidate_nodes: Sequence[Dict[str, Any]],
    report_language: str = "en",
) -> OptimizationReport:
    """Build a useful deterministic report if the report agent fails."""
    is_en = _normalize_report_language(report_language) == "en"
    groups = _fallback_strategy_groups(report_language)
    designs = []
    for index, node in enumerate(candidate_nodes, start=1):
        group_id = _fallback_group_for_candidate(node, index)
        group_count = 1 + sum(1 for item in designs if item.strategy_group == group_id)
        designs.append(_candidate_to_design(node, f"{group_id}{group_count}", group_id, report_language))

    return OptimizationReport(
        report_title="SAR Optimization Design Report" if is_en else "SAR 改造设计报告",
        target_summary=optimization_goal,
        original_compound_label="Root compound" if is_en else "原始化合物",
        core_scaffold=(
            "Summarize property and SAR directions around the final MCGS candidate nodes, starting from the input scaffold."
            if is_en else "基于输入原始化合物结构，围绕 MCGS 最终候选节点进行性质与 SAR 方向总结。"
        ),
        key_structural_features=(
            [
                "Treat the root scaffold and any user-protected region as the primary pharmacophore hypothesis.",
                "Use candidate actions, critic rationales, and properties to infer modification strategies.",
                "Prioritize local changes that improve the target property without substantially weakening drug-like balance.",
            ]
            if is_en else [
                "原始化合物骨架和用户限定保留区域应作为主要药效团假设。",
                "最终候选分子的 action、critic rationale 和 properties 用于推断改造策略。",
                "优先关注可改善目标性质且不显著破坏类药性的局部改造。",
            ]
        ),
        recommended_modification_sites=_fallback_modification_sites(candidate_nodes, report_language),
        strategy_groups=groups,
        designed_molecules=designs,
        first_round_synthesis=_fallback_first_round(designs, candidate_nodes, report_language=report_language),
        risk_notes=_fallback_risk_notes(optimization_goal, report_language),
    )


def build_report_html(
    report: OptimizationReport,
    *,
    initial_smiles: str,
    initial_iupac: Any,
    initial_fragments: Any,
    optimization_goal: str,
    candidate_nodes: Sequence[Dict[str, Any]],
    report_language: str = "en",
) -> str:
    parent_props = calculate_molecule_properties(initial_smiles)
    parent_image = molecule_png_base64(
        initial_smiles,
        size=(420, 336),
        asset_name="fallback_parent_topology",
    )
    candidate_by_smiles = {node.get("smiles"): node for node in candidate_nodes}

    groups = _sort_groups(report.strategy_groups)
    designs_by_group = _group_designs(report.designed_molecules, groups)
    today = datetime.now().strftime("%Y-%m-%d")
    language = _normalize_report_language(report_language)
    is_en = language == "en"
    initial_iupac_text = _normalize_molecule_fact_value(
        initial_iupac,
        initial_smiles,
        fact_key="iupac",
    ) or "None"
    initial_fragments_text = _normalize_molecule_fact_value(
        initial_fragments,
        initial_smiles,
        fact_key="fragments",
    ) or "None"

    parts = [
        f"<!doctype html><html lang=\"{language}\"><head><meta charset=\"utf-8\"/>",
        f"<title>{_e(report.report_title)}</title>",
        _style_block(),
        "</head><body>",
        f"<h1>{_e(report.report_title)}</h1>",
        (
            f"<p style=\"color:#9fb0bf\">Objective: {_e(report.target_summary or optimization_goal)}. Date: {today}.</p>"
            if is_en
            else f"<p style=\"color:#9fb0bf\">目标：{_e(report.target_summary or optimization_goal)}。日期：{today}。</p>"
        ),
        f"<h2>1. {'Root Molecule' if is_en else '原始化合物'} {_e(report.original_compound_label)}</h2>",
        "<div class=\"parentbox\">",
        _image_tag(parent_image, "420", "336"),
        "<div>",
        _property_tags(parent_props),
        "<div class=\"mol-facts\">",
        f"<div><b>SMILES</b><span class=\"mono\">{_e(initial_smiles)}</span></div>",
        f"<div><b>IUPAC</b><span>{_e(initial_iupac_text)}</span></div>",
        f"<div><b>Fragments</b><span>{_e(initial_fragments_text)}</span></div>",
        "</div>",
        (
            f"<p style=\"margin-top:10px\"><b>Core scaffold</b>: {_e(report.core_scaffold)}</p>"
            if is_en
            else f"<p style=\"margin-top:10px\"><b>核心骨架</b>：{_e(report.core_scaffold)}</p>"
        ),
        "<p><b>Main pharmacophore/features</b>:</p>" if is_en else "<p><b>主要药效团元素</b>：</p>",
        "<ul class=\"feature-list\">",
        *[f"<li>{_e(feature)}</li>" for feature in report.key_structural_features],
        "</ul>",
        "</div></div>",
        "<h2>2. Recommended Modification Sites</h2>" if is_en else "<h2>2. 推荐改造位点</h2>",
        "<ol>",
        *[
            f"<li><b>{_e(site.site_id)} {'(' if is_en else '（'}{_e(site.site_name)}{')' if is_en else '）'}</b>{':' if is_en else '：'} {_e(site.recommendation)}</li>"
            for site in report.recommended_modification_sites
        ],
        "</ol>",
        (
            "<h2>3. Strategy Groups and Design Logic</h2>"
            if is_en
            else "<h2>3. 改造策略与设计逻辑（按组）</h2>"
        ),
        "<p>Strategy summary by group:</p>" if is_en else "<p>每组设计原则简述如下：</p>",
        (
            "<table class=\"summary\"><tr><th>Group</th><th>Design purpose</th><th>SAR output</th></tr>"
            if is_en
            else "<table class=\"summary\"><tr><th>组别</th><th>设计目的</th><th>SAR 输出</th></tr>"
        ),
        *[
            "<tr>"
            f"<td>{_e(group.group_id)} {_e(group.group_name)}</td>"
            f"<td>{_e(group.design_purpose)}</td>"
            f"<td>{_e(group.sar_output)}</td>"
            "</tr>"
            for group in groups
        ],
        "</table>",
    ]

    for group in groups:
        group_designs = designs_by_group.get(group.group_id, [])
        if not group_designs:
            continue
        parts.append(f"<h2>4. {_e(group.group_id)} {_e(group.group_name)}</h2><div class='cards'>")
        for design in group_designs:
            parts.append(_render_design_card(design, group, parent_props, candidate_by_smiles, initial_smiles, language))
        parts.append("</div>")

    parts.extend(
        [
            "<h2>5. Priority and First-Round Synthesis List</h2>" if is_en else "<h2>5. 优先级与第一轮合成清单</h2>",
            _render_priority_summary(report.designed_molecules, language),
            "<h2>6. First Round Recommended Synthesis</h2>" if is_en else "<h2>6. 第一轮（建议合成）</h2>",
            _render_first_round_table(report.first_round_synthesis, report.designed_molecules, language),
            "<h2>7. Risks and Notes</h2>" if is_en else "<h2>7. 风险与注意事项</h2>",
            "<ul>",
            *[f"<li>{_e(note)}</li>" for note in report.risk_notes],
            "</ul>",
            (
                "<p class=\"legend\">Note: cLogP/MW are RDKit calculations (MolLogP, MolWt). Delta values are relative to the root molecule.</p>"
                if is_en
                else "<p class=\"legend\">注：所有 cLogP/MW 由 RDKit 计算（MolLogP, MolWt）。Δ 为相对原始化合物。</p>"
            ),
            "</body></html>",
        ]
    )
    return "\n".join(parts)


def _coerce_report(report_content: Optional[Any]) -> Optional[OptimizationReport]:
    if report_content is None:
        return None
    if isinstance(report_content, OptimizationReport):
        return report_content
    try:
        return OptimizationReport.model_validate(report_content)
    except (ValidationError, TypeError, ValueError):
        return None


def _candidate_to_design(node: Dict[str, Any], design_id: str, group_id: str, report_language: str = "en") -> ReportDesignedMolecule:
    is_en = _normalize_report_language(report_language) == "en"
    action = node.get("action") or node.get("modification_type") or (
        "Candidate modification" if is_en else "候选分子改造"
    )
    critic = node.get("critic_rationale") or ""
    generator = node.get("generator_rationale") or ""
    score = _float_or_none(node.get("score"))
    priority = "H" if score is not None and score >= 0.75 else "M" if score is not None and score >= 0.5 else "L"
    return ReportDesignedMolecule(
        design_id=design_id,
        smiles=node.get("smiles", ""),
        strategy_group=group_id,
        design_logic=_compact_text(
            action, generator,
            default=(
                "Local structural modification proposed by the MCGS search path."
                if is_en else "基于 MCGS 搜索路径提出的局部结构改造。"
            ),
        ),
        sar_exploration_direction=_compact_text(
            critic,
            default=(
                "Validate the effect on the target property, activity retention, and overall drug-like balance."
                if is_en else "验证该改造对目标性质、活性保持和整体类药性的影响。"
            ),
        ),
        synthetic_feasibility=(
            "Assess late-stage derivatization or fragment replacement around the parent scaffold, including available building blocks and routine couplings."
            if is_en else "基于母核进行后期衍生化或片段替换，建议优先评估市售砌块与常规偶联可行性。"
        ),
        priority=priority,
    )


def _normalize_report_language(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"en", "eng", "english"}:
        return "en"
    if text in {"zh", "cn", "chs", "chinese", "中文", "汉语"}:
        return "zh"
    return "zh"


def _contains_cjk(text: Any) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(text or "")))


def _resolve_report_language(
    report_language: Optional[str],
    *,
    report_request_text: str,
    state: Dict[str, Any],
) -> str:
    explicit = str(report_language or "").strip()
    if explicit:
        return _normalize_report_language(explicit)
    text_parts = [
        report_request_text,
        state.get("report_request_text", ""),
        state.get("last_report_request", ""),
        state.get("user_prompt", ""),
        state.get("optimization_goal", ""),
        state.get("project_manager_brief", ""),
    ]
    combined = "\n".join(str(part or "") for part in text_parts)
    if _contains_cjk(combined):
        return "zh"
    if re.search(r"[A-Za-z]{3,}", combined):
        return "en"
    return "zh"


def _resolve_max_detailed_cards(
    candidate_count: int,
    *,
    max_detailed_cards: Optional[int],
    top_n: int,
    first_round_n: int,
) -> int:
    count = max(0, int(candidate_count or 0))
    if count <= 0:
        return 0
    explicit = _int_or_none(max_detailed_cards)
    if explicit is not None and explicit > 0:
        return min(count, explicit)
    if count <= 30:
        return count
    return min(count, max(20, int(top_n or 0), int(first_round_n or 0)))


def _extract_admet_policy(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "selected_admet_properties": state.get("selected_admet_properties") or [],
        "selected_admet_preference_json": state.get("selected_admet_preference_json") or "{}",
        "selected_admet_property_details": state.get("selected_admet_property_details") or [],
        "selected_admet_property_rationale": state.get("selected_admet_property_rationale") or "",
        "selected_admet_task_signature": state.get("selected_admet_task_signature") or "",
    }


def _normalize_blueprint_groups(raw_groups: Sequence[Any], language: str) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw in enumerate(raw_groups):
        group = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw or {})
        group_id = str(group.get("group_id") or "").strip().upper()
        if not group_id:
            group_id = _alpha_group_id(index)
        if group_id in used_ids:
            continue
        fallback = _fallback_blueprint_group(group_id, language)
        groups.append(
            {
                "group_id": group_id,
                "group_name": str(group.get("group_name") or fallback["group_name"]).strip(),
                "design_purpose": str(group.get("design_purpose") or fallback["design_purpose"]).strip(),
                "sar_output": str(group.get("sar_output") or fallback["sar_output"]).strip(),
                "section_intro": str(group.get("section_intro") or fallback["section_intro"]).strip(),
            }
        )
        used_ids.add(group_id)
    return groups


def _fallback_blueprint_groups(language: str) -> List[Dict[str, Any]]:
    return [_fallback_blueprint_group(group_id, language) for group_id in ("A", "B", "C")]


def _fallback_blueprint_group(group_id: str, language: str) -> Dict[str, str]:
    gid = str(group_id or "A").strip().upper()[:2] or "A"
    if _normalize_report_language(language) == "en":
        names = {
            "A": "High-scoring graph candidates",
            "B": "Property-balance analogs",
            "C": "Exploratory graph designs",
        }
        purposes = {
            "A": "Prioritize reachable candidates with stronger graph scores and clear structural changes.",
            "B": "Probe polarity, lipophilicity, hydrogen bonding, and size balance around the root scaffold.",
            "C": "Preserve structurally diverse MCGS proposals as SAR exploration points.",
        }
        outputs = {
            "A": "Tests whether the dominant graph direction improves the requested objective.",
            "B": "Links RDKit property shifts and supplied rationales to qualitative SAR trends.",
            "C": "Identifies whether adjacent chemical space deserves another optimization round.",
        }
        return {
            "group_id": gid,
            "group_name": names.get(gid, f"Strategy {gid} candidates"),
            "design_purpose": purposes.get(gid, "Document a strategy group represented by the current graph."),
            "sar_output": outputs.get(gid, "Clarify SAR value, feasibility, and risk for this strategy."),
            "section_intro": "This group is organized from the current MCGS graph and uses only supplied node evidence.",
        }
    names = {"A": "高分候选优化", "B": "性质平衡优化", "C": "探索性改造"}
    purposes = {
        "A": "优先梳理评分较高、结构变化清晰且可达的图候选。",
        "B": "围绕极性、脂溶性、氢键模式和分子量平衡建立 SAR 梯度。",
        "C": "保留 MCGS 提供的多样化探索方向，判断是否值得进入下一轮。",
    }
    outputs = {
        "A": "验证主要图搜索方向是否有望改善用户目标。",
        "B": "关联 RDKit 理化性质变化、已有 rationale 与定性 SAR 信号。",
        "C": "识别相邻结构空间的潜在价值、可行性和风险。",
    }
    return {
        "group_id": gid,
        "group_name": names.get(gid, f"策略 {gid} 候选"),
        "design_purpose": purposes.get(gid, "记录当前 MCGS 图中出现的策略组。"),
        "sar_output": outputs.get(gid, "明确该策略的 SAR 价值、合成可行性和主要风险。"),
        "section_intro": "本组完全基于当前 MCGS 图节点、action、rationale 和计算属性整理。",
    }


def _fallback_blueprint_card(
    node: Dict[str, Any],
    *,
    asset_by_smiles: Dict[str, Dict[str, Any]],
    design_id: str,
    group_id: str,
    language: str,
) -> Dict[str, Any]:
    smiles = str(node.get("smiles") or "").strip()
    asset = asset_by_smiles.get(smiles, {})
    score = _float_or_none(node.get("score"))
    priority = "H" if score is not None and score >= 0.75 else "M" if score is not None and score >= 0.5 else "L"
    if bool(node.get("unreachable")) and priority == "H":
        priority = "M"
    selected_reason = node.get("selected_reason")
    if isinstance(selected_reason, (list, tuple)):
        selected_reason_text = " ".join(str(item) for item in selected_reason if item)
    else:
        selected_reason_text = str(selected_reason or "")
    if _normalize_report_language(language) == "en":
        return {
            "design_id": design_id,
            "asset_id": asset.get("asset_id") or "",
            "smiles": smiles,
            "strategy_group": group_id,
            "priority": priority,
            "role_tags": ["auto-completed"] + (["unreachable"] if node.get("unreachable") else []),
            "design_logic": _compact_text(
                node.get("action", ""),
                node.get("generator_rationale", ""),
                default="Graph-derived local modification proposed by the MCGS workflow.",
            ),
            "sar_exploration_direction": _compact_text(
                node.get("critic_rationale", ""),
                selected_reason_text,
                default="Validate the expected effect on the requested objective and core drug-like balance.",
            ),
            "synthetic_feasibility": "Assess route availability before synthesis; treat this as a graph-derived design hypothesis rather than a confirmed route.",
        }
    return {
        "design_id": design_id,
        "asset_id": asset.get("asset_id") or "",
        "smiles": smiles,
        "strategy_group": group_id,
        "priority": priority,
        "role_tags": ["自动补齐"] + (["unreachable"] if node.get("unreachable") else []),
        "design_logic": _compact_text(
            node.get("action", ""),
            node.get("generator_rationale", ""),
            default="基于 MCGS 搜索路径提出的局部结构改造。",
        ),
        "sar_exploration_direction": _compact_text(
            node.get("critic_rationale", ""),
            selected_reason_text,
            default="建议验证该改造对用户目标和整体类药性平衡的影响。",
        ),
        "synthetic_feasibility": "合成前需确认路线可得性；当前仅作为图搜索得到的设计假设，而非已确认路线。",
    }


def _next_blueprint_design_id(group_id: str, used_design_ids: set[str]) -> str:
    prefix = str(group_id or "M").strip().upper()[:2] or "M"
    index = 1
    while f"{prefix}{index}" in used_design_ids:
        index += 1
    return f"{prefix}{index}"


def _normalize_blueprint_first_round(
    raw_first_round: Sequence[Any],
    cards: Sequence[Dict[str, Any]],
    *,
    node_by_smiles: Dict[str, Dict[str, Any]],
    first_round_n: int,
    language: str,
) -> List[Dict[str, str]]:
    limit = max(0, int(first_round_n or 0))
    if limit <= 0:
        return []
    card_by_id = {str(card.get("design_id") or ""): card for card in cards if card.get("design_id")}
    result: List[Dict[str, str]] = []
    seen_ids: set[str] = set()
    for raw in raw_first_round:
        item = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw or {})
        design_id = str(item.get("design_id") or "").strip()
        card = card_by_id.get(design_id)
        if not card or design_id in seen_ids:
            continue
        node = node_by_smiles.get(str(card.get("smiles") or "").strip(), {})
        if bool(node.get("unreachable")):
            continue
        reason = str(item.get("reason") or "").strip() or _first_round_default_reason(language)
        result.append({"design_id": design_id, "reason": reason})
        seen_ids.add(design_id)
        if len(result) >= limit:
            return result

    sorted_cards = sorted(
        cards,
        key=lambda card: (
            {"H": 0, "M": 1, "L": 2}.get(str(card.get("priority") or "L"), 3),
            -(_float_or_none(node_by_smiles.get(str(card.get("smiles") or ""), {}).get("score")) or 0.0),
            str(card.get("design_id") or ""),
        ),
    )
    for card in sorted_cards:
        design_id = str(card.get("design_id") or "").strip()
        smiles = str(card.get("smiles") or "").strip()
        if not design_id or design_id in seen_ids or bool(node_by_smiles.get(smiles, {}).get("unreachable")):
            continue
        result.append({"design_id": design_id, "reason": _first_round_default_reason(language)})
        seen_ids.add(design_id)
        if len(result) >= limit:
            break
    return result


def _first_round_default_reason(language: str) -> str:
    if _normalize_report_language(language) == "en":
        return "High SAR value in the current graph, with clear structural change and suitable first-round validation value."
    return "在当前图中具有较高 SAR 信息量，结构变化清晰，适合作为第一轮验证点。"


def _fallback_blueprint_risk_notes(language: str) -> List[str]:
    if _normalize_report_language(language) == "en":
        return [
            "All recommendations are graph-derived hypotheses and require experimental validation.",
            "Confirm target activity, requested ADMET endpoints, solubility, and synthetic feasibility before prioritizing scale-up.",
            "Treat unreachable graph nodes as audit observations, not first-round synthesis recommendations.",
        ]
    return [
        "所有建议均为基于当前图和已有证据的设计假设，需要实验证实。",
        "第一轮需同步确认目标活性、用户指定 ADMET 端点、溶解度和合成可行性。",
        "unreachable 图节点仅作为审计记录，不应作为第一轮高优先级合成建议。",
    ]


def _fallback_blueprint_expert_questions(language: str) -> List[str]:
    if _normalize_report_language(language) == "en":
        return [
            "Which assay readout should gate the next optimization round?",
            "Are any scaffold or substituent changes synthetically disallowed by the project team?",
            "Which ADMET risk should override graph score when priorities conflict?",
        ]
    return [
        "下一轮优化应以哪个实验读数作为主要门槛？",
        "项目团队是否有禁止改动的骨架或取代基区域？",
        "当图评分与 ADMET 风险冲突时，应优先规避哪一类风险？",
    ]


def _alpha_group_id(index: int) -> str:
    if 0 <= index < 26:
        return chr(ord("A") + index)
    return f"G{index + 1}"


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fallback_strategy_groups(report_language: str = "en") -> List[ReportStrategyGroup]:
    return [
        ReportStrategyGroup(**{
            key: value
            for key, value in _fallback_blueprint_group(group_id, report_language).items()
            if key in {"group_id", "group_name", "design_purpose", "sar_output"}
        })
        for group_id in ("A", "B", "C")
    ]


def _fallback_group_for_candidate(node: Dict[str, Any], index: int) -> str:
    if node.get("unreachable"):
        return "C"
    score = _float_or_none(node.get("score"))
    if score is not None and score >= 0.72:
        return "A"
    if index % 3 == 0:
        return "C"
    return "B"


def _fallback_modification_sites(
    candidate_nodes: Sequence[Dict[str, Any]], report_language: str = "en"
) -> List[ReportModificationSite]:
    actions = [str(node.get("action") or "") for node in candidate_nodes[:8] if node.get("action")]
    is_en = _normalize_report_language(report_language) == "en"
    action_hint = (
        ("; " if is_en else "；").join(actions[:3])
        if actions else (
            "Locally modify high-scoring candidates according to their recorded actions"
            if is_en else "结合最终图中高分候选的 action 进行局部改造"
        )
    )
    if is_en:
        return [
            ReportModificationSite(
                site_id="Site 1",
                site_name="User-permitted or frequently explored variable region",
                recommendation=f"Build a local SAR series around this region; representative changes include: {action_hint}.",
            ),
            ReportModificationSite(
                site_id="Site 2",
                site_name="Polarity/lipophilicity adjustment point",
                recommendation="Balance the target property and drug-like profile through bioisosteric replacement, smaller hydrophobic groups, or mild polarity additions.",
            ),
            ReportModificationSite(
                site_id="Site 3",
                site_name="Noncritical substituent near the core scaffold",
                recommendation="Test small substituents and conformational constraints without disrupting the core pharmacophore.",
            ),
        ]
    return [
        ReportModificationSite(
            site_id="位点 1",
            site_name="用户允许或搜索集中出现的可变片段",
            recommendation=f"优先围绕该区域开展 SAR 梯度；代表性改造包括：{action_hint}。",
        ),
        ReportModificationSite(
            site_id="位点 2",
            site_name="影响目标性质的极性/脂溶性调节点",
            recommendation="通过等排、缩小疏水体积或引入温和极性原子来平衡目标性质与类药性。",
        ),
        ReportModificationSite(
            site_id="位点 3",
            site_name="核心骨架周边非关键取代",
            recommendation="在不破坏核心药效团的前提下验证小取代基和构象约束对 SAR 的贡献。",
        ),
    ]


def _fallback_first_round(
    designs: Sequence[Any],
    candidate_nodes: Sequence[Dict[str, Any]],
    limit: int = 10,
    report_language: str = "en",
) -> List[ReportFirstRoundCandidate]:
    candidate_score = {node.get("smiles"): _float_or_none(node.get("score")) for node in candidate_nodes}
    design_dicts = [
        item.model_dump() if isinstance(item, ReportDesignedMolecule) else dict(item)
        for item in designs
    ]
    design_dicts.sort(
        key=lambda item: (
            {"H": 0, "M": 1, "L": 2}.get(item.get("priority"), 3),
            -(candidate_score.get(item.get("smiles")) or 0.0),
        )
    )
    result = []
    for item in design_dicts[:limit]:
        result.append(
            ReportFirstRoundCandidate(
                design_id=item.get("design_id", ""),
                reason=(
                    "Higher priority and clear structural change make this suitable for first-round SAR validation."
                    if _normalize_report_language(report_language) == "en"
                    else "优先级较高，结构变化清晰，适合作为第一轮 SAR 验证点。"
                ),
            )
        )
    return result


def _fallback_risk_notes(optimization_goal: str, report_language: str = "en") -> List[str]:
    if _normalize_report_language(report_language) == "en":
        return [
            f"Validate every design experimentally against the objective '{optimization_goal}'; current conclusions are computational and medicinal-chemistry hypotheses.",
            "In the first round, assess primary activity, requested ADMET/PK risk metrics, and baseline physicochemical properties together.",
            "For basic, strongly hydrophobic, or highly planar fragments, also monitor hERG, CYP, solubility, and nonspecific binding risks.",
            "Before synthesis, confirm route availability, protecting-group compatibility, and key intermediate supply for high-priority molecules.",
        ]
    return [
        f"所有设计均需围绕目标“{optimization_goal}”进行实测验证，当前判断为计算与药化假设。",
        "第一轮建议同步评估主要活性读数、目标 ADMET/PK 风险指标和基础理化性质。",
        "若引入碱性、强疏水或高平面性片段，应额外关注 hERG、CYP、溶解度和非特异结合风险。",
        "合成前建议对高优先级分子进行路线可得性、保护基兼容性和关键中间体供应确认。",
    ]


def _sort_groups(groups: Sequence[ReportStrategyGroup]) -> List[ReportStrategyGroup]:
    return sorted(groups, key=lambda group: group.group_id)


def _group_designs(
    designs: Sequence[ReportDesignedMolecule],
    groups: Sequence[ReportStrategyGroup],
) -> Dict[str, List[ReportDesignedMolecule]]:
    group_ids = {group.group_id for group in groups}
    grouped: Dict[str, List[ReportDesignedMolecule]] = {group.group_id: [] for group in groups}
    fallback_group = groups[-1].group_id if groups else "Z"
    for design in designs:
        group_id = _resolve_group_id(design.strategy_group, group_ids) or fallback_group
        grouped.setdefault(group_id, []).append(design)
    for items in grouped.values():
        items.sort(key=lambda item: ({"H": 0, "M": 1, "L": 2}.get(item.priority, 3), item.design_id))
    return grouped


def _resolve_group_id(value: str, group_ids: Iterable[str]) -> Optional[str]:
    if value in group_ids:
        return value
    prefix = (value or "").strip()[:1]
    return prefix if prefix in group_ids else None


def _render_design_card(
    design: ReportDesignedMolecule,
    group: ReportStrategyGroup,
    parent_props: Dict[str, Any],
    candidate_by_smiles: Dict[str, Dict[str, Any]],
    initial_smiles: str,
    report_language: str = "en",
) -> str:
    is_en = _normalize_report_language(report_language) == "en"
    props = calculate_molecule_properties(design.smiles)
    asset_key = hashlib.sha1((design.design_id + "|" + design.smiles).encode("utf-8")).hexdigest()[:10]
    image = molecule_png_base64(
        design.smiles,
        asset_name=f"fallback_{design.design_id}_{asset_key}_topology",
    )
    diff_image, diff_summary = molecule_comparison_png_base64(
        initial_smiles,
        design.smiles,
        size=(760, 320),
        asset_name=f"fallback_{design.design_id}_{asset_key}_diff",
    )
    candidate = candidate_by_smiles.get(design.smiles, {})
    score = candidate.get("score")
    score_text = (
        f"<div class=\"row\"><b>MCGS score{':' if is_en else '：'}</b>{_e(_fmt_number(score))}</div>"
        if score is not None else ""
    )
    return "\n".join(
        [
            f"<div class=\"card prio-{_e(design.priority)}\">",
            "<div class=\"cardhead\">",
            f"<span class=\"did\">{_e(design.design_id)}</span>",
            f"<span class=\"grp\">{_e(group.group_id)} {_e(group.group_name)}</span>",
            f"<span class=\"prio\">{'Priority' if is_en else '优先级'} {_e(design.priority)}</span>",
            "</div>",
            '<div class="mol-visuals">',
            '<div class="topologybox">',
            '<div class="visual-title">Candidate 2D topology</div>' if is_en else '<div class="visual-title">候选二维拓扑</div>',
            _image_tag(image, "320", "260"),
            "</div>",
            '<div class="diffbox">',
            '<div class="visual-title">Changes versus root molecule</div>' if is_en else '<div class="visual-title">与初始分子比变化高亮</div>',
            _image_tag(diff_image, "760", "320"),
            _visual_legend_html(report_language=report_language),
            f'<p class="diff-summary">{_e(_format_diff_summary(diff_summary, report_language=report_language))}</p>',
            "</div>",
            "</div>",
            f"<div class=\"smi\">{_e(design.smiles)}</div>",
            _property_table(props, parent_props),
            f"<div class=\"row\"><b>{'Design logic:' if is_en else '设计逻辑：'}</b>{_e(design.design_logic)}</div>",
            f"<div class=\"row\"><b>{'SAR hypothesis:' if is_en else 'SAR 假设：'}</b>{_e(design.sar_exploration_direction)}</div>",
            f"<div class=\"row\"><b>{'Synthetic feasibility:' if is_en else '合成可行性：'}</b>{_e(design.synthetic_feasibility)}</div>",
            score_text,
            "</div>",
        ]
    )


def _render_priority_summary(
    designs: Sequence[ReportDesignedMolecule], report_language: str = "en"
) -> str:
    rows = []
    for design in sorted(designs, key=lambda item: ({"H": 0, "M": 1, "L": 2}.get(item.priority, 3), item.design_id)):
        rows.append(
            "<tr>"
            f"<td>{_e(design.design_id)}</td>"
            f"<td>{_e(design.priority)}</td>"
            f"<td>{_e(design.strategy_group)}</td>"
            f"<td>{_e(design.sar_exploration_direction)}</td>"
            "</tr>"
        )
    is_en = _normalize_report_language(report_language) == "en"
    return (
        (
            "<table class=\"summary\"><tr><th>ID</th><th>Priority</th><th>Group</th><th>SAR direction</th></tr>"
            if is_en else "<table class=\"summary\"><tr><th>ID</th><th>优先级</th><th>组别</th><th>SAR 探索方向</th></tr>"
        )
        + "\n".join(rows)
        + "</table>"
    )


def _render_first_round_table(
    first_round: Sequence[ReportFirstRoundCandidate],
    designs: Sequence[ReportDesignedMolecule],
    report_language: str = "en",
) -> str:
    design_by_id = {design.design_id: design for design in designs}
    rows = []
    for item in first_round:
        design = design_by_id.get(item.design_id)
        rows.append(
            "<tr>"
            f"<td>{_e(item.design_id)}</td>"
            f"<td>{_e(design.smiles if design else '')}</td>"
            f"<td>{_e(item.reason)}</td>"
            "</tr>"
        )
    header = (
        "<table class=\"summary\"><tr><th>ID</th><th>SMILES</th><th>Reason</th></tr>"
        if _normalize_report_language(report_language) == "en"
        else "<table class=\"summary\"><tr><th>ID</th><th>SMILES</th><th>推荐理由</th></tr>"
    )
    return header + "\n".join(rows) + "</table>"


def _property_tags(props: Dict[str, Any]) -> str:
    return "\n".join(
        f"<div class=\"tag\">{_e(col)} {_e(_format_property(props, col))}</div>"
        for col in PROPERTY_COLUMNS
    )


def _property_table(props: Dict[str, Any], parent_props: Dict[str, Any]) -> str:
    headers = "".join(f"<th>{_e(col)}</th>" for col in PROPERTY_COLUMNS)
    cells = "".join(f"<td>{_property_cell(props, parent_props, col)}</td>" for col in PROPERTY_COLUMNS)
    return f"<table class=\"ptable\"><tr>{headers}</tr><tr>{cells}</tr></table>"


def _property_cell(props: Dict[str, Any], parent_props: Dict[str, Any], col: str) -> str:
    value = _format_property(props, col)
    if col in {"MW", "cLogP"} and props.get(col) is not None and parent_props.get(col) is not None:
        delta = props[col] - parent_props[col]
        return f"{_e(value)} <span class=\"d\">({_e(_signed(delta, 1 if col == 'MW' else 2))})</span>"
    return _e(value)


def _format_property(props: Dict[str, Any], col: str) -> str:
    value = props.get(col)
    if col == "Ro5":
        return "✓" if value else "✗"
    if value is None:
        return "N/A"
    return str(value)


def _image_tag(image_base64: str, width: str, height: str) -> str:
    if not image_base64:
        return (
            f"<div style=\"width:{width}px;height:{height}px;border:1px solid #eee;"
            "display:flex;align-items:center;justify-content:center;color:#888\">Invalid SMILES</div>"
        )
    return f"<img src=\"data:image/png;base64,{image_base64}\"/>"


def _compact_text(*parts: str, default: str) -> str:
    text = " ".join(str(part).strip() for part in parts if part and str(part).strip())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return default
    return text[:320]


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_number(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return str(value)
    return f"{number:.3f}"


def _signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}"


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _resolve_initial_report_facts(state: MCGSState, mol_graph: Any) -> Tuple[str, str, str]:
    root_node = getattr(mol_graph, "root_node", None) if mol_graph else None
    root_smiles = str(getattr(root_node, "smiles", "") or "").strip()
    initial_smiles = str(state.get("initial_smiles") or root_smiles or "").strip()

    facts = _lookup_molecule_facts(state.get("molecule_facts_cache"), initial_smiles)
    if not facts and root_smiles:
        facts = _lookup_molecule_facts(state.get("molecule_facts_cache"), root_smiles)
    if not facts:
        facts = _lookup_current_info_facts(state.get("current_info_list"), initial_smiles)

    initial_iupac = _normalize_molecule_fact_value(
        state.get("initial_iupac"),
        initial_smiles,
        fact_key="iupac",
    )
    initial_fragments = _normalize_molecule_fact_value(
        state.get("initial_fragments"),
        initial_smiles,
        fact_key="fragments",
    )

    if _is_missing_report_fact(initial_iupac):
        initial_iupac = _normalize_molecule_fact_value(
            facts.get("iupac") if facts else None,
            initial_smiles,
            fact_key="iupac",
        )
    if _is_missing_report_fact(initial_fragments):
        initial_fragments = _normalize_molecule_fact_value(
            facts.get("fragments") if facts else None,
            initial_smiles,
            fact_key="fragments",
        )

    return initial_smiles, initial_iupac, initial_fragments


def _lookup_molecule_facts(cache: Any, smiles: str) -> Dict[str, Any]:
    if not isinstance(cache, dict) or not smiles:
        return {}

    lookup_keys = [key for key in (smiles, _canonical_report_smiles_key(smiles)) if key]
    for key in lookup_keys:
        value = cache.get(key)
        if isinstance(value, dict):
            return value

    target_key = _canonical_report_smiles_key(smiles)
    for key, value in cache.items():
        if not isinstance(value, dict):
            continue
        fact_smiles = str(value.get("smiles") or key or "").strip()
        if fact_smiles == smiles:
            return value
        if target_key and _canonical_report_smiles_key(fact_smiles) == target_key:
            return value
    return {}


def _lookup_current_info_facts(current_info_list: Any, smiles: str) -> Dict[str, Any]:
    if not isinstance(current_info_list, list) or not current_info_list:
        return {}

    target_key = _canonical_report_smiles_key(smiles)
    first_valid: Dict[str, Any] = {}
    for item in current_info_list:
        if not isinstance(item, dict):
            continue
        item_smiles = str(item.get("current_smiles") or item.get("smiles") or "").strip()
        facts = {
            "smiles": item_smiles,
            "iupac": item.get("current_iupac") or item.get("iupac"),
            "fragments": item.get("current_fragments") or item.get("fragments"),
        }
        if not first_valid and item_smiles:
            first_valid = facts
        if item_smiles == smiles:
            return facts
        if target_key and _canonical_report_smiles_key(item_smiles) == target_key:
            return facts
    return first_valid if not smiles else {}


def _normalize_molecule_fact_value(value: Any, smiles: str, fact_key: Optional[str] = None) -> str:
    if _is_missing_report_fact(value):
        return ""

    if isinstance(value, dict):
        if fact_key and fact_key in value:
            return _normalize_molecule_fact_value(value.get(fact_key), smiles)

        for key in (smiles, _canonical_report_smiles_key(smiles)):
            if key and key in value:
                return _normalize_molecule_fact_value(value.get(key), smiles)

        target_key = _canonical_report_smiles_key(smiles)
        if target_key:
            for key, item in value.items():
                if _canonical_report_smiles_key(key) == target_key:
                    return _normalize_molecule_fact_value(item, smiles)

        if len(value) == 1:
            return _normalize_molecule_fact_value(next(iter(value.values())), smiles)

        for key in ("value", "text", "result", "content"):
            if key in value:
                return _normalize_molecule_fact_value(value.get(key), smiles)

        return json.dumps(value, ensure_ascii=False, default=str)

    if isinstance(value, (list, tuple, set)):
        items = [
            item
            for item in (_normalize_molecule_fact_value(item, smiles) for item in value)
            if not _is_missing_report_fact(item)
        ]
        return ", ".join(items)

    text = str(value).strip()
    return "" if _is_missing_report_fact(text) else text


def _is_missing_report_fact(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return not value
    if isinstance(value, (list, tuple, set)):
        return not value
    text = str(value).strip()
    return not text or text.lower() in {"none", "unknown", "n/a", "na", "null", "{}", "[]"}


def _canonical_report_smiles_key(smiles: Any) -> str:
    text = str(smiles or "").strip()
    if not text:
        return ""
    try:
        mol, canonical, error = parse_smiles_quiet(text)
    except Exception:
        return ""
    if mol is None or error:
        return ""
    return canonical or text


def _style_block() -> str:
    return """<style>
	body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,"Microsoft YaHei",sans-serif;margin:0;color:#d8e2ec;background:#081017;padding:28px;max-width:1720px}
	h1{margin:0 0 10px;color:#f3f7fb;letter-spacing:0}
	h2{border-left:4px solid #51c7a9;padding-left:12px;margin-top:34px;color:#eef6fb}
	.parentbox{background:#101820;border:1px solid #284052;border-radius:8px;padding:16px;display:flex;gap:24px;align-items:flex-start;box-shadow:0 22px 60px rgba(0,0,0,.25)}
	.parentbox img{border:1px solid #314655;border-radius:6px;background:#fff}
	.mol-facts{margin-top:8px;border:1px solid #293c4c;border-radius:6px;background:#0e1720;overflow:hidden}
	.mol-facts div{display:grid;grid-template-columns:90px minmax(0,1fr);gap:8px;border-top:1px solid #293c4c;padding:7px 9px;font-size:12.5px}
	.mol-facts div:first-child{border-top:0}
	.mol-facts b{color:#dfeaf4}
	.mol-facts span{word-break:break-word}
	.mol-facts .mono{font-family:Consolas,monospace;color:#e6edf3}
	.tag{display:inline-block;background:#182838;border:1px solid #2b455a;border-radius:10px;padding:2px 10px;margin:2px 4px 2px 0;font-size:12px;color:#dce8f3}
	.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}
.card{background:#101820;border:1px solid #284052;border-radius:8px;padding:12px;box-shadow:0 18px 45px rgba(0,0,0,.22)}
.card img{display:block;margin:0 auto}
.card .smi{font-family:Consolas,monospace;font-size:11px;color:#e6edf3;word-break:break-all;margin:6px 0}
.card .row{font-size:12.5px;margin:5px 0;line-height:1.45;color:#cbd8e3}
.cardhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.did{font-weight:700;color:#63d7bd}
.grp{font-size:11px;color:#9fb0bf}
.prio{font-size:11px;background:#243240;border:1px solid #3b5368;border-radius:8px;padding:1px 8px;color:#dce8f3}
.prio-H .prio{background:#123b32;border-color:#2fb28e;color:#c8fff1}
.prio-M .prio{background:#3e3518;border-color:#b5922b;color:#fff0ba}
.prio-L .prio{background:#27313b;border-color:#465767;color:#ced8e2}
.ptable{width:100%;border-collapse:collapse;margin:6px 0;font-size:11.5px}
.ptable th,.ptable td{border:1px solid #293c4c;padding:3px 4px;text-align:center}
.ptable th{background:#172433;color:#dfeaf4}
.ptable .d{color:#9fb0bf;font-size:10px}
table.summary{border-collapse:collapse;width:100%;margin-top:8px}
table.summary th,table.summary td{border:1px solid #263847;padding:7px;font-size:13px;vertical-align:top}
table.summary th{background:#172433;color:#dfeaf4}
table.summary td{background:#0e1720;color:#d4e0ea}
.feature-list li{margin:4px 0}
.legend{font-size:12px;color:#9fb0bf;margin-top:8px}
""" + _report_visual_css() + """
@media(max-width:760px){body{margin:12px}.parentbox{display:block}.parentbox img{max-width:100%;height:auto}.cards{grid-template-columns:1fr}}
</style>"""
