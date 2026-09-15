"""End-to-end Kimi Code report generation for M3OS search graphs.

The molecular-search workflow owns the data; Kimi Code owns the report coding
task.  This module therefore exports one versioned JSON handoff, starts Kimi
Code in a report-specific workspace, and accepts the generated HTML as the
deliverable.  It deliberately does not call the legacy LangChain ReportAgent or
its blueprint/HTML post-processing pipeline.
"""

from __future__ import annotations

import asyncio
import copy
import html
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from bs4 import BeautifulSoup

from m3os.agents_v5.core.config import AgentConfig, PROJECT_ROOT, PROJECT_TMP_DIR

if TYPE_CHECKING:
    from m3os.agents_v5.core.state import MCGSState


REPORT_HANDOFF_SCHEMA_VERSION = "m3os-kimi-code-report-v2"
REPORT_SKILL_ROOT = PROJECT_ROOT / "m3os" / "agents_v5" / "skills"
REPORT_SKILL_DIR = REPORT_SKILL_ROOT / "kimi-code-search-report"
REPORT_ASSET_RENDERER = REPORT_SKILL_DIR / "render_molecule_assets.py"
REPORT_GRAPH_RENDERER = REPORT_SKILL_DIR / "render_search_graph_component.py"
REPORT_GRAPH_COMPONENT_FILENAME = "molecular_search_graph_component.html"
LANDLOCK_LAUNCHER = PROJECT_ROOT / "m3os" / "agents_v5" / "services" / "landlock_exec.py"
LOCAL_KIMI_CODE_BINARY = PROJECT_TMP_DIR / "kimi-code-cli" / "bin" / "kimi"
DEFAULT_KIMI_CODE_BINARY = Path.home() / ".kimi-code" / "bin" / "kimi"
DEFAULT_REPORT_WORKSPACE_ROOT = (
    Path.home() / ".local" / "share" / "m3os" / "report-workspaces"
)
KIMI_RUNTIME_CONFIG_FILES = (
    "config.toml",
    "tui.toml",
    "region",
    "device_id",
    "migrations-effort.json",
)


@dataclass
class KimiCodeReportResult:
    """Files and runtime metadata produced by one Kimi Code report run."""

    html: str
    output_path: str
    workspace_path: str
    input_path: str
    assets_path: str
    graph_component_path: str
    stdout_path: str
    stderr_path: str
    mode: str = "kimi_code"
    cli_returncode: int = 0


class KimiCodeReportGenerator:
    """Give a complete M3OS graph to Kimi Code as an HTML coding task."""

    def __init__(self, config: AgentConfig):
        self.config = config

    async def generate(
        self,
        *,
        graph_snapshot: Mapping[str, Any],
        state: Optional[MCGSState] = None,
        output_dir: str | Path = "tmp/reports",
        filename: Optional[str] = None,
        top_n: int = 8,
        first_round_n: Optional[int] = None,
        coverage_mode: str = "all",
        max_detailed_cards: Optional[int] = None,
        style_reference: str = "m3os_evolution_atlas",
        report_language: Optional[str] = None,
        report_request_text: str = "",
        synthesis_shortlist_n: Optional[int] = None,
    ) -> KimiCodeReportResult:
        """Run a fresh Kimi Code session and return its complete HTML report."""
        safe_snapshot = self._assign_molecule_numbers(
            self._enrich_graph_snapshot(graph_snapshot, state)
        )
        nodes = [item for item in safe_snapshot.get("nodes", []) if isinstance(item, dict)]
        candidate_nodes = [item for item in nodes if not item.get("is_root")]
        root_node_ids = {
            str(item.get("id") or "") for item in nodes if item.get("is_root")
        }
        if not candidate_nodes:
            raise ValueError("The current molecular search graph has no candidate nodes to report.")

        command = self._resolve_kimi_command()
        output_root = self._resolve_output_dir(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        workspace = self._resolve_workspace_root() / run_id
        workspace.mkdir(parents=True, exist_ok=False)
        self._stage_report_skill(workspace)

        output_name = self._normalize_output_filename(filename, run_id=run_id)
        workspace_report = workspace / "report.html"
        final_report = output_root / output_name
        input_path = workspace / "report_input.json"
        assets_path = workspace / "molecule_assets.json"
        graph_component_path = workspace / REPORT_GRAPH_COMPONENT_FILENAME
        stdout_path = workspace / "kimi_stdout.jsonl"
        stderr_path = workspace / "kimi_stderr.log"

        payload = self._build_handoff_payload(
            graph_snapshot=safe_snapshot,
            state=state,
            output_filename=workspace_report.name,
            top_n=top_n,
            synthesis_shortlist_n=(
                synthesis_shortlist_n
                if synthesis_shortlist_n is not None
                else (first_round_n if first_round_n is not None else 5)
            ),
            coverage_mode=coverage_mode,
            max_detailed_cards=max_detailed_cards,
            style_reference=style_reference,
            report_language=report_language,
            report_request_text=report_request_text,
        )
        self._write_json(input_path, payload)
        await self._render_molecule_assets(input_path=input_path, assets_path=assets_path)
        await self._render_search_graph_component(
            input_path=input_path,
            assets_path=assets_path,
            output_path=graph_component_path,
        )

        # Kimi Code may need more than one pass to satisfy the report contract.
        # Keep these generated inputs authoritative across repair attempts: the
        # model is allowed to rewrite its report/builder, but not the evidence or
        # the prebuilt graph component that the validator relies on.
        authoritative_files = {
            input_path: input_path.read_bytes(),
            assets_path: assets_path.read_bytes(),
            graph_component_path: graph_component_path.read_bytes(),
        }

        timeout_seconds = self._positive_float_env(
            "KIMI_CODE_REPORT_TIMEOUT_SECONDS",
            self.config.llm.kimi_request_timeout_seconds,
        )
        repair_attempts = self._nonnegative_int_env(
            "KIMI_CODE_REPORT_REPAIR_ATTEMPTS",
            2,
        )
        repair_backoff_seconds = self._nonnegative_float_env(
            "KIMI_CODE_REPORT_REPAIR_BACKOFF_SECONDS",
            5.0,
        )
        expected_graph_edges = [
            item
            for item in payload["search_graph"].get("edges", [])
            if isinstance(item, Mapping)
        ]
        expected_graph_component_html = graph_component_path.read_text(
            encoding="utf-8"
        ).rstrip("\n")
        expected_molecule_numbers = {
            str(node_id): int(number)
            for node_id, number in payload["molecule_numbering"][
                "node_id_to_number"
            ].items()
        }

        failures: list[str] = []
        html_text: Optional[str] = None
        returncode = -1
        for attempt_index in range(repair_attempts + 1):
            if attempt_index == 0:
                prompt = self._build_prompt(
                    input_filename=input_path.name,
                    output_filename=workspace_report.name,
                )
                attempt_stdout_path = stdout_path
                attempt_stderr_path = stderr_path
            else:
                if repair_backoff_seconds:
                    await asyncio.sleep(
                        repair_backoff_seconds * (2 ** (attempt_index - 1))
                    )
                prompt = self._build_repair_prompt(
                    input_filename=input_path.name,
                    output_filename=workspace_report.name,
                    failure=failures[-1],
                    repair_attempt=attempt_index,
                    repair_attempts=repair_attempts,
                )
                attempt_stdout_path = (
                    workspace / f"kimi_repair_{attempt_index}_stdout.jsonl"
                )
                attempt_stderr_path = (
                    workspace / f"kimi_repair_{attempt_index}_stderr.log"
                )

            try:
                returncode, stdout, stderr = await self._run_kimi_code(
                    command=command,
                    workspace=workspace,
                    prompt=prompt,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                returncode = -1
                stdout = ""
                stderr = f"{type(exc).__name__}: {exc}"

            attempt_stdout_path.write_text(
                stdout,
                encoding="utf-8",
                errors="replace",
            )
            attempt_stderr_path.write_text(
                stderr,
                encoding="utf-8",
                errors="replace",
            )

            changed_inputs = self._restore_authoritative_files(authoritative_files)
            if changed_inputs:
                failure = (
                    "Kimi Code modified authoritative report inputs; they were restored: "
                    + ", ".join(changed_inputs)
                )
            elif workspace_report.exists():
                candidate_html = workspace_report.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                try:
                    self._validate_report_html(
                        candidate_html,
                        candidate_nodes=candidate_nodes,
                        root_node_ids=root_node_ids,
                        expected_graph_edges=expected_graph_edges,
                        expected_graph_component_html=expected_graph_component_html,
                        expected_language=str(payload["report_request"]["language"]),
                        expected_detailed_cards=int(
                            payload["report_request"]["detailed_card_limit"]
                        ),
                        expected_optimization_round_count=int(
                            payload["search_graph_semantics"][
                                "optimization_round_count"
                            ]
                        ),
                        expected_synthesis_count=min(
                            int(
                                payload["report_request"][
                                    "synthesis_shortlist_count"
                                ]
                            ),
                            len(candidate_nodes),
                        ),
                        expected_molecule_numbers=expected_molecule_numbers,
                    )
                except Exception as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    if returncode != 0:
                        failure += (
                            "; Kimi Code also exited unsuccessfully: "
                            f"exit_code={returncode}; "
                            f"stderr_tail={self._tail(stderr)}"
                        )
                else:
                    html_text = candidate_html
                    stdout_path = attempt_stdout_path
                    stderr_path = attempt_stderr_path
                    break
            elif returncode != 0:
                failure = (
                    "Kimi Code report generation exited unsuccessfully. "
                    f"exit_code={returncode}; stderr_tail={self._tail(stderr)}"
                )
            else:
                failure = (
                    "Kimi Code did not create report.html. "
                    f"exit_code={returncode}; stderr_tail={self._tail(stderr)}"
                )

            failures.append(f"attempt {attempt_index + 1}: {failure}")

        if html_text is None:
            failure_summary = " | ".join(self._tail(item) for item in failures)
            raise RuntimeError(
                "Kimi Code could not produce a valid report after "
                f"{repair_attempts + 1} attempts; workspace={workspace}; "
                f"failures={failure_summary}"
            )

        shutil.copyfile(workspace_report, final_report)
        return KimiCodeReportResult(
            html=html_text,
            output_path=str(final_report),
            workspace_path=str(workspace),
            input_path=str(input_path),
            assets_path=str(assets_path),
            graph_component_path=str(graph_component_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            cli_returncode=returncode,
        )

    def _resolve_kimi_command(self) -> Path:
        configured = str(os.getenv("KIMI_CODE_COMMAND") or "").strip()
        candidates: list[Path] = []
        if configured:
            candidates.append(Path(configured).expanduser())
        discovered = shutil.which("kimi")
        if discovered:
            candidates.append(Path(discovered))
        candidates.extend((DEFAULT_KIMI_CODE_BINARY, LOCAL_KIMI_CODE_BINARY))

        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            if resolved.is_file() and os.access(resolved, os.X_OK):
                return resolved
        raise FileNotFoundError(
            "Kimi Code CLI was not found. Install `kimi`, set KIMI_CODE_COMMAND, "
            f"or install the project-local binary at {LOCAL_KIMI_CODE_BINARY}."
        )

    @staticmethod
    def _resolve_output_dir(output_dir: str | Path) -> Path:
        path = Path(output_dir).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @staticmethod
    def _resolve_workspace_root() -> Path:
        configured = str(os.getenv("KIMI_CODE_REPORT_WORKSPACE_ROOT") or "").strip()
        path = Path(configured).expanduser() if configured else DEFAULT_REPORT_WORKSPACE_ROOT
        resolved = path.resolve()
        try:
            resolved.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            pass
        else:
            raise ValueError(
                "KIMI_CODE_REPORT_WORKSPACE_ROOT must be outside the M3OS project tree."
            )
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @staticmethod
    def _stage_report_skill(workspace: Path) -> None:
        destination = workspace / "skills" / REPORT_SKILL_DIR.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            REPORT_SKILL_DIR,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    @staticmethod
    async def _render_molecule_assets(*, input_path: Path, assets_path: Path) -> None:
        timeout = float(os.getenv("KIMI_CODE_REPORT_ASSET_TIMEOUT_SECONDS") or 300)

        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [
                    sys.executable,
                    str(REPORT_ASSET_RENDERER),
                    "--input",
                    str(input_path),
                    "--output",
                    str(assets_path),
                ],
                cwd=str(input_path.parent),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )

        result = await asyncio.to_thread(_run)
        if result.returncode != 0 or not assets_path.is_file():
            raise RuntimeError(
                "Failed to build parent-aligned molecule difference assets. "
                f"exit_code={result.returncode}; stderr={result.stderr[-3000:]}"
            )

    @staticmethod
    async def _render_search_graph_component(
        *, input_path: Path, assets_path: Path, output_path: Path
    ) -> None:
        timeout = float(os.getenv("KIMI_CODE_REPORT_GRAPH_TIMEOUT_SECONDS") or 120)

        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [
                    sys.executable,
                    str(REPORT_GRAPH_RENDERER),
                    "--input",
                    str(input_path),
                    "--assets",
                    str(assets_path),
                    "--output",
                    str(output_path),
                ],
                cwd=str(input_path.parent),
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )

        result = await asyncio.to_thread(_run)
        if result.returncode != 0 or not output_path.is_file():
            raise RuntimeError(
                "Failed to render the prebuilt Molecular Search Graph component. "
                f"exit_code={result.returncode}; stderr={result.stderr[-3000:]}"
            )

    @staticmethod
    def _normalize_output_filename(filename: Optional[str], *, run_id: str) -> str:
        proposed = Path(str(filename or f"m3os_kimi_code_report_{run_id}.html")).name
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(proposed).stem).strip("._")
        return f"{stem or 'm3os_kimi_code_report'}.html"

    def _build_handoff_payload(
        self,
        *,
        graph_snapshot: Mapping[str, Any],
        state: Optional[MCGSState],
        output_filename: str,
        top_n: int,
        synthesis_shortlist_n: int,
        coverage_mode: str,
        max_detailed_cards: Optional[int],
        style_reference: str,
        report_language: Optional[str],
        report_request_text: str,
    ) -> dict[str, Any]:
        state_map: Mapping[str, Any] = state or {}
        language = self._resolve_language(report_language, report_request_text, state_map)
        graph_nodes = [
            item for item in graph_snapshot.get("nodes", []) if isinstance(item, Mapping)
        ]
        iteration_values: list[int] = []
        for item in graph_nodes:
            try:
                iteration_values.append(int(item.get("iteration", 0)))
            except (TypeError, ValueError):
                continue
        optimization_round_count = max(
            (iteration for iteration in iteration_values if iteration > 0),
            default=0,
        )
        candidate_count = sum(1 for item in graph_nodes if not item.get("is_root"))
        requested_card_limit = (
            max(1, int(max_detailed_cards))
            if max_detailed_cards is not None
            else max(1, int(top_n))
        )
        detailed_card_limit = min(candidate_count, requested_card_limit)
        node_id_to_number = {
            str(item.get("id") or ""): int(item["molecule_number"])
            for item in graph_nodes
        }
        return {
            "schema_version": REPORT_HANDOFF_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "task": {
                "user_prompt": str(state_map.get("user_prompt") or ""),
                "project_manager_brief": str(state_map.get("project_manager_brief") or ""),
                "optimization_goal": str(state_map.get("optimization_goal") or ""),
                "initial_smiles": str(
                    state_map.get("initial_smiles")
                    or graph_snapshot.get("root_smiles")
                    or ""
                ),
                "initial_iupac": self._json_safe(state_map.get("initial_iupac") or ""),
                "initial_fragments": self._json_safe(state_map.get("initial_fragments") or ""),
                "selected_admet_properties": self._json_safe(
                    state_map.get("selected_admet_properties") or []
                ),
                "selected_admet_preference_json": str(
                    state_map.get("selected_admet_preference_json") or ""
                ),
                "selected_admet_property_details": self._json_safe(
                    state_map.get("selected_admet_property_details") or []
                ),
                "selected_admet_property_rationale": str(
                    state_map.get("selected_admet_property_rationale") or ""
                ),
                "current_info_list": self._json_safe(state_map.get("current_info_list") or []),
            },
            "search_graph": self._json_safe(graph_snapshot),
            "molecule_numbering": {
                "label_format": "Molecule {number}",
                "starts_at": 1,
                "authoritative_index": "Full-Node Audit",
                "ordering": (
                    "Root first; then iteration ascending, score descending, "
                    "and stable node ID ascending."
                ),
                "node_id_to_number": node_id_to_number,
                "node_id_to_label": {
                    node_id: f"Molecule {number}"
                    for node_id, number in node_id_to_number.items()
                },
            },
            "search_graph_semantics": {
                "root_iteration": 0,
                "iteration_levels_including_root": sorted(set(iteration_values)),
                "optimization_round_count": optimization_round_count,
                "definition": (
                    "Iteration 0 is the root baseline and is not an optimization round."
                ),
            },
            "report_request": {
                "language": language,
                "coverage_mode": str(coverage_mode or "all"),
                "top_n_focus": max(1, int(top_n)),
                "synthesis_shortlist_count": max(0, int(synthesis_shortlist_n)),
                "max_detailed_cards": (
                    max(1, int(max_detailed_cards))
                    if max_detailed_cards is not None
                    else None
                ),
                "detailed_card_limit": detailed_card_limit,
                "molecule_label_preference": "molecule_number",
                "molecule_number_label_format": "Molecule {number}",
                "show_full_smiles_without_truncation": True,
                "full_node_audit_format": "table",
                "full_node_audit_is_molecule_index": True,
                "audit_table_topology_thumbnails": True,
                "candidate_descriptor_layout": "wrapped_rows",
                "candidate_grid_desktop_columns": 3,
                "recommendation_layout": "stacked_full_width",
                "recommendation_tier_columns": 3,
                "synthesis_shortlist_format": "full_width_table",
                "evolution_graph_component": "m3os-molecular-search-graph-v1",
                "prebuilt_graph_component": REPORT_GRAPH_COMPONENT_FILENAME,
                "graph_component_policy": "embed-verbatim-exactly-once",
                "evolution_graph_layout": "layered-by-generation-and-parent",
                "evolution_graph_interactions": [
                    "select-node-details",
                    "large-pointer-hit-target",
                    "zoom",
                    "pan",
                    "drag-node",
                    "fit",
                ],
                "style_direction": str(style_reference or "m3os_evolution_atlas"),
                "user_report_request": str(report_request_text or ""),
            },
            "coding_runtime": {
                "input_file": "report_input.json",
                "molecule_assets_file": "molecule_assets.json",
                "graph_component_file": REPORT_GRAPH_COMPONENT_FILENAME,
                "required_output_file": output_filename,
                "filesystem_policy": (
                    "This disposable report workspace is the only readable/writable task data. "
                    "The M3OS project tree and private development references are not exposed."
                ),
            },
            "source_of_truth": [
                "The search_graph nodes and edges are the complete molecular search result for this report.",
                "Iteration 0 is the root baseline; use search_graph_semantics for optimization-round counts.",
                "Candidate SMILES must be copied exactly from search_graph.nodes; never invent or rewrite a molecule.",
                "Use each node's supplied molecule_number and molecule_label everywhere; never renumber molecules in the report.",
                "The Full-Node Audit is the authoritative molecule-number lookup and must contain every graph node exactly once.",
                "Scores and predicted properties are computational evidence, not experimental measurements.",
                "When evidence is missing, label the point as an uncertainty or proposed validation.",
            ],
        }

    @classmethod
    def _enrich_graph_snapshot(
        cls,
        graph_snapshot: Mapping[str, Any],
        state: Optional[MCGSState],
    ) -> dict[str, Any]:
        """Preserve every parent-specific graph edge when the live graph is available."""
        snapshot = cls._json_safe(copy.deepcopy(dict(graph_snapshot)))
        if not state or not state.get("mcgs_graph"):
            return snapshot

        graph = state["mcgs_graph"]
        nodes = list(graph.get_all_nodes()) if hasattr(graph, "get_all_nodes") else []
        score_by_id = {
            str(getattr(node, "id", "") or ""): cls._number_or_none(
                getattr(node, "intrinsic_score", None)
            )
            for node in nodes
        }
        edges: list[dict[str, Any]] = []
        for child in nodes:
            child_id = str(getattr(child, "id", "") or "")
            child_score = cls._number_or_none(getattr(child, "intrinsic_score", None))
            parents = list(graph.get_parents(child)) if hasattr(graph, "get_parents") else []
            for parent in parents:
                parent_id = str(getattr(parent, "id", "") or "")
                parent_smiles = str(getattr(parent, "smiles", "") or "")
                keys = [key for key in (parent_id, parent_smiles) if key]
                parent_score = score_by_id.get(parent_id)
                edges.append(
                    {
                        "source": parent_id,
                        "target": child_id,
                        "source_smiles": parent_smiles,
                        "target_smiles": str(getattr(child, "smiles", "") or ""),
                        "action": str(
                            cls._first_value(getattr(child, "actions_from_parents", {}), keys)
                            or ""
                        ),
                        "generator_rationale": str(
                            cls._first_value(
                                getattr(child, "rationale_from_parents_generator", {}),
                                keys,
                            )
                            or ""
                        ),
                        "critic_rationale": str(
                            cls._first_value(
                                getattr(child, "rationale_from_parents_critic", {}),
                                keys,
                            )
                            or ""
                        ),
                        "generator_confidence": cls._number_or_none(
                            cls._first_value(
                                getattr(child, "confidence_score_from_parents_generator", {}),
                                keys,
                            )
                        ),
                        "score_delta": (
                            child_score - parent_score
                            if child_score is not None and parent_score is not None
                            else None
                        ),
                    }
                )
        if edges:
            snapshot["edges"] = cls._json_safe(edges)
        return snapshot

    @classmethod
    def _assign_molecule_numbers(
        cls, graph_snapshot: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Attach stable, human-facing molecule numbers to every graph node."""
        snapshot = cls._json_safe(copy.deepcopy(dict(graph_snapshot)))
        nodes = [
            item
            for item in (snapshot.get("nodes") or [])
            if isinstance(item, dict)
        ]
        node_ids = [str(item.get("id") or "") for item in nodes]
        if not nodes:
            return snapshot
        if any(not node_id for node_id in node_ids):
            raise ValueError("Every graph node must have a non-empty id before numbering.")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Graph node IDs must be unique before molecule numbering.")

        def _iteration(node: Mapping[str, Any]) -> int:
            try:
                return int(node.get("iteration") or 0)
            except (TypeError, ValueError):
                return 0

        def _sort_key(node: Mapping[str, Any]) -> tuple[Any, ...]:
            score = cls._number_or_none(node.get("score"))
            return (
                0 if node.get("is_root") else 1,
                _iteration(node),
                1 if score is None else 0,
                -(score or 0.0),
                str(node.get("id") or ""),
            )

        ordered_nodes = sorted(nodes, key=_sort_key)
        number_by_id = {
            str(node["id"]): index
            for index, node in enumerate(ordered_nodes, start=1)
        }
        for node in nodes:
            molecule_number = number_by_id[str(node["id"])]
            node["molecule_number"] = molecule_number
            node["molecule_label"] = f"Molecule {molecule_number}"

        for edge in snapshot.get("edges", []):
            if not isinstance(edge, dict):
                continue
            source_id = str(edge.get("source") or "")
            target_id = str(edge.get("target") or "")
            if source_id in number_by_id:
                edge["source_molecule_number"] = number_by_id[source_id]
                edge["source_molecule_label"] = (
                    f"Molecule {number_by_id[source_id]}"
                )
            if target_id in number_by_id:
                edge["target_molecule_number"] = number_by_id[target_id]
                edge["target_molecule_label"] = (
                    f"Molecule {number_by_id[target_id]}"
                )

        snapshot["molecule_index"] = [
            {
                "molecule_number": number_by_id[str(node["id"])],
                "molecule_label": f"Molecule {number_by_id[str(node['id'])]}",
                "node_id": str(node["id"]),
                "smiles": str(node.get("smiles") or ""),
                "is_root": bool(node.get("is_root")),
                "iteration": _iteration(node),
            }
            for node in ordered_nodes
        ]
        return snapshot

    @staticmethod
    def _first_value(values: Any, preferred_keys: Sequence[str]) -> Any:
        if not isinstance(values, Mapping) or not values:
            return ""
        for key in preferred_keys:
            if key in values:
                return values[key]
        return next(iter(values.values()))

    @staticmethod
    def _number_or_none(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number != number or number in {float("inf"), float("-inf")}:
            return None
        return number

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return cls._number_or_none(value)
        if isinstance(value, Mapping):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return cls._json_safe(model_dump())
        return str(value)

    @staticmethod
    def _resolve_language(
        requested: Optional[str],
        report_request_text: str,
        state: Mapping[str, Any],
    ) -> str:
        normalized = str(requested or "").strip().lower()
        if normalized in {"en", "english"}:
            return "en"
        if normalized in {"zh", "cn", "chinese", "中文"}:
            return "zh"
        combined = " ".join(
            [
                str(report_request_text or ""),
                str(state.get("user_prompt") or ""),
                str(state.get("project_manager_brief") or ""),
            ]
        )
        return "zh" if re.search(r"[\u4e00-\u9fff]", combined) else "en"

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _build_prompt(*, input_filename: str, output_filename: str) -> str:
        return (
            "Load the `kimi-code-search-report` Skill. Read "
            f"`{input_filename}` and `molecule_assets.json`. Reuse the complete prebuilt "
            f"`{REPORT_GRAPH_COMPONENT_FILENAME}` verbatim without reading or rewriting "
            "its implementation, then create and validate "
            f"the final report at exactly `{output_filename}`. The files in this "
            "workspace are the complete task input; do not invent missing evidence."
        )

    @staticmethod
    def _build_repair_prompt(
        *,
        input_filename: str,
        output_filename: str,
        failure: str,
        repair_attempt: int,
        repair_attempts: int,
    ) -> str:
        return (
            "Load the `kimi-code-search-report` Skill and repair the existing report. "
            f"This is repair pass {repair_attempt} of {repair_attempts}. The previous "
            "report attempt failed the application check with this exact result:\n\n"
            f"{KimiCodeReportGenerator._tail(failure, max_chars=6000)}\n\n"
            f"Treat `{input_filename}`, `molecule_assets.json`, and "
            f"`{REPORT_GRAPH_COMPONENT_FILENAME}` as immutable authoritative inputs. "
            "Do not rewrite them. Inspect the existing report.html and any report "
            "builder source in the workspace, correct the specific failure and any "
            "closely related issue, execute the builder if one exists, and validate "
            f"the result. Write the corrected final report at exactly `{output_filename}`. "
            "Preserve the requested report language and do not invent missing evidence."
        )

    @staticmethod
    def _restore_authoritative_files(files: Mapping[Path, bytes]) -> list[str]:
        restored: list[str] = []
        for path, expected_bytes in files.items():
            try:
                current_bytes = path.read_bytes()
            except (FileNotFoundError, IsADirectoryError, OSError):
                current_bytes = None
            if current_bytes == expected_bytes and not path.is_symlink():
                continue
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.exists():
                shutil.rmtree(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected_bytes)
            restored.append(path.name)
        return restored

    @staticmethod
    def _prepare_kimi_runtime_home(
        *,
        command: Path,
        workspace: Path,
        directory_name: str = ".kimi-runtime",
    ) -> Path:
        configured_home = str(os.getenv("KIMI_CODE_HOME") or "").strip()
        source_home = (
            Path(configured_home).expanduser()
            if configured_home
            else DEFAULT_KIMI_CODE_BINARY.parent.parent
        )
        if not (source_home / "config.toml").is_file():
            fallback = command.parent.parent
            if (fallback / "config.toml").is_file():
                source_home = fallback
        if not (source_home / "config.toml").is_file():
            raise FileNotFoundError(
                "Kimi Code config.toml was not found for the isolated report session."
            )

        runtime_home = workspace / directory_name
        runtime_home.mkdir(mode=0o700)
        for filename in KIMI_RUNTIME_CONFIG_FILES:
            source = source_home / filename
            if source.is_file():
                destination = runtime_home / filename
                shutil.copyfile(source, destination)
                destination.chmod(0o600)
        return runtime_home

    @staticmethod
    def _sandbox_argv(*, command: Path, workspace: Path, kimi_argv: Sequence[str]) -> list[str]:
        read_paths = [
            Path("/usr"),
            Path("/bin"),
            Path("/lib"),
            Path("/lib64"),
            Path("/etc"),
            Path("/proc"),
            Path("/sys"),
            Path("/run"),
            command.parent,
        ]
        write_paths = [workspace, Path("/tmp"), Path("/var/tmp"), Path("/dev")]
        argv = [sys.executable, str(LANDLOCK_LAUNCHER)]
        for path in read_paths:
            if path.exists():
                argv.extend(("--read", str(path.resolve())))
        for path in write_paths:
            if path.exists():
                argv.extend(("--write", str(path.resolve())))
        argv.append("--")
        argv.extend(kimi_argv)
        return argv

    async def _run_kimi_code(
        self,
        *,
        command: Path,
        workspace: Path,
        prompt: str,
        timeout_seconds: float,
    ) -> tuple[int, str, str]:
        env = dict(os.environ)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        # Do not let an unrelated temporary-provider override replace the
        # user's installed Kimi Code provider/login.  Report-specific runtime
        # controls are set explicitly below without rewriting config.toml.
        for name in (
            "KIMI_MODEL_NAME",
            "KIMI_MODEL_API_KEY",
            "KIMI_MODEL_PROVIDER_TYPE",
            "KIMI_MODEL_BASE_URL",
            "KIMI_MODEL_MAX_CONTEXT_SIZE",
            "KIMI_MODEL_CAPABILITIES",
            "KIMI_MODEL_DISPLAY_NAME",
        ):
            env.pop(name, None)
        env["KIMI_DISABLE_TELEMETRY"] = "1"
        effort = str(
            os.getenv("KIMI_CODE_REPORT_REASONING_EFFORT")
            or self.config.llm.kimi_reasoning_effort
            or "low"
        ).strip().lower()
        if effort not in {"low", "high", "max"}:
            raise ValueError("KIMI_CODE_REPORT_REASONING_EFFORT must be low, high, or max.")
        env["KIMI_MODEL_THINKING_EFFORT"] = effort
        env["KIMI_LOOP_MAX_STEPS_PER_TURN"] = str(
            self._positive_int_env("KIMI_CODE_REPORT_MAX_STEPS", 40)
        )
        env["KIMI_LOOP_MAX_ATTEMPTS_PER_STEP"] = str(
            self._positive_int_env(
                "KIMI_CODE_REPORT_MAX_ATTEMPTS_PER_STEP",
                self.config.llm.kimi_max_retries,
            )
        )

        session_token = uuid.uuid4().hex[:12]
        runtime_home = self._prepare_kimi_runtime_home(
            command=command,
            workspace=workspace,
            directory_name=f".kimi-runtime-{session_token}",
        )
        isolated_home = workspace / f".kimi-home-{session_token}"
        isolated_tmp = workspace / f".kimi-tmp-{session_token}"
        try:
            isolated_home.mkdir(mode=0o700)
            isolated_tmp.mkdir(mode=0o700)
            env["KIMI_CODE_HOME"] = str(runtime_home)
            env["HOME"] = str(isolated_home)
            env["TMPDIR"] = str(isolated_tmp)

            kimi_argv = [str(command)]
            model_alias = str(os.getenv("KIMI_CODE_REPORT_MODEL") or "").strip()
            if model_alias:
                kimi_argv.extend(("--model", model_alias))
            kimi_argv.extend([
                "--skills-dir",
                str((workspace / "skills").resolve()),
                "--prompt",
                prompt,
                "--output-format",
                "stream-json",
            ])
            argv = self._sandbox_argv(
                command=command,
                workspace=workspace,
                kimi_argv=kimi_argv,
            )
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(workspace),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                await self._terminate_process_group(process)
                raise TimeoutError(
                    f"Kimi Code report generation exceeded {timeout_seconds:.0f} seconds; "
                    f"workspace={workspace}"
                ) from exc
        finally:
            shutil.rmtree(runtime_home, ignore_errors=True)
            shutil.rmtree(isolated_home, ignore_errors=True)
            shutil.rmtree(isolated_tmp, ignore_errors=True)
        return (
            int(process.returncode or 0),
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
        )

    @staticmethod
    async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            await process.wait()

    @staticmethod
    def _validate_report_html(
        html_text: str,
        *,
        candidate_nodes: Sequence[Mapping[str, Any]],
        root_node_ids: Sequence[str] = (),
        expected_graph_edges: Optional[Sequence[Mapping[str, Any]]] = None,
        expected_graph_component_html: Optional[str] = None,
        expected_language: Optional[str] = None,
        expected_detailed_cards: Optional[int] = None,
        expected_optimization_round_count: Optional[int] = None,
        expected_synthesis_count: Optional[int] = None,
        expected_molecule_numbers: Optional[Mapping[str, int]] = None,
    ) -> None:
        lowered = html_text.lower()
        required_tokens = ("<!doctype html", "<html", "<head", "<style", "<body")
        missing_tokens = [token for token in required_tokens if token not in lowered]
        if missing_tokens:
            raise ValueError(
                "Kimi Code report is not a complete self-contained HTML document; "
                f"missing: {', '.join(missing_tokens)}"
            )
        if len(html_text.encode("utf-8")) < 10_000:
            raise ValueError("Kimi Code report is unexpectedly small (<10 KB).")
        if "\ufffd" in html_text:
            raise ValueError(
                "Kimi Code report contains Unicode replacement characters; "
                "the generated copy is corrupted."
            )
        if lowered.count("<code") != lowered.count("</code>"):
            raise ValueError("Kimi Code report contains unbalanced <code> tags.")

        soup = BeautifulSoup(html_text, "html.parser")
        if soup.find("title") is None or not soup.find("title").get_text(strip=True):
            raise ValueError("Kimi Code report has no non-empty <title>.")
        normalized_language = str(expected_language or "").strip().lower()
        html_tag = soup.find("html")
        document_language = str(html_tag.get("lang") or "").strip().lower() if html_tag else ""
        if normalized_language == "zh":
            if not document_language.startswith("zh"):
                raise ValueError(
                    "Kimi Code report ignored the requested Chinese language: "
                    f"html lang={document_language or 'missing'}."
                )
            title_text = soup.find("title").get_text(" ", strip=True)
            if not re.search(r"[\u4e00-\u9fff]", title_text):
                raise ValueError("Kimi Code report has a non-Chinese title for a Chinese request.")
            visible_soup = BeautifulSoup(html_text, "html.parser")
            # Validate localization on the surrounding report narrative rather
            # than letting the graph's full-node inspector dominate the ratio.
            for graph in visible_soup.find_all(
                attrs={"data-evolution-graph": "m3os-molecular-search-graph-v1"}
            ):
                graph.decompose()
            for tag in visible_soup.find_all(("script", "style", "svg", "code")):
                tag.decompose()
            visible_text = visible_soup.get_text(" ", strip=True)
            cjk_count = len(re.findall(r"[\u4e00-\u9fff]", visible_text))
            latin_count = len(re.findall(r"[A-Za-z]", visible_text))
            # Molecular reports legitimately contain long SMILES plus supplied
            # English action/rationale evidence.  Require substantial Chinese
            # UI copy without treating those source-of-truth strings as a
            # localization failure.
            if cjk_count < 200 or cjk_count / max(1, cjk_count + latin_count) < 0.15:
                raise ValueError(
                    "Kimi Code report is not predominantly Chinese despite the language request."
                )

        if expected_optimization_round_count is not None:
            round_meta = soup.find(
                "meta", attrs={"name": "m3os-optimization-round-count"}
            )
            actual_round_count = str(round_meta.get("content") or "") if round_meta else ""
            if actual_round_count != str(expected_optimization_round_count):
                raise ValueError(
                    "Kimi Code report has a missing or incorrect optimization-round marker: "
                    f"expected {expected_optimization_round_count}, got {actual_round_count or 'missing'}."
                )

        graph_section = soup.find(
            attrs={"data-evolution-graph": "m3os-molecular-search-graph-v1"}
        )
        if graph_section is None:
            raise ValueError(
                "Kimi Code report must use the M3OS Molecular Search Graph component."
            )
        if expected_graph_component_html is not None:
            component_text = expected_graph_component_html.rstrip("\n")
            if html_text.count(component_text) != 1:
                raise ValueError(
                    "Kimi Code report must embed the prebuilt Molecular Search Graph "
                    "component verbatim exactly once."
                )
        required_graph_hooks = (
            "data-graph-metrics",
            "data-graph-inspector",
            "data-graph-stage",
            "data-graph-controls",
            "data-graph-svg",
            "data-graph-panzoom",
            "data-graph-legend",
            "data-graph-node-details",
        )
        missing_graph_hooks = [
            hook
            for hook in required_graph_hooks
            if graph_section.find(attrs={hook: True}) is None
        ]
        if missing_graph_hooks:
            raise ValueError(
                "Kimi Code report Molecular Search Graph is missing component hooks: "
                + ", ".join(missing_graph_hooks)
            )
        if not graph_section.find_all(attrs={"data-graph-layer": True}):
            raise ValueError(
                "Kimi Code report Molecular Search Graph has no generation layer bands."
            )
        for selector in (
            {"data-graph-zoom": "in"},
            {"data-graph-zoom": "out"},
            {"data-graph-fit": True},
        ):
            if graph_section.find(attrs=selector) is None:
                raise ValueError(
                    "Kimi Code report Molecular Search Graph lacks zoom or Fit controls."
                )

        expected_graph_node_ids = {
            str(node.get("id") or "") for node in candidate_nodes
        } | {str(node_id) for node_id in root_node_ids if str(node_id)}
        molecule_number_by_id = {
            str(node_id): int(number)
            for node_id, number in (expected_molecule_numbers or {}).items()
        }
        if set(molecule_number_by_id) != expected_graph_node_ids:
            raise ValueError(
                "The expected molecule-number mapping does not cover every graph node."
            )
        if sorted(molecule_number_by_id.values()) != list(
            range(1, len(expected_graph_node_ids) + 1)
        ):
            raise ValueError(
                "Molecule numbers must be unique and contiguous, starting at 1."
            )

        def molecule_label(node_id: str) -> str:
            return f"Molecule {molecule_number_by_id[node_id]}"

        graph_node_ids = [
            str(tag.get("data-graph-node") or "")
            for tag in graph_section.find_all(attrs={"data-graph-node": True})
        ]
        graph_detail_ids = [
            str(tag.get("data-graph-node-detail") or "")
            for tag in graph_section.find_all(attrs={"data-graph-node-detail": True})
        ]
        if (
            len(graph_node_ids) != len(expected_graph_node_ids)
            or len(set(graph_node_ids)) != len(graph_node_ids)
            or set(graph_node_ids) != expected_graph_node_ids
        ):
            raise ValueError(
                "Kimi Code report Molecular Search Graph must render every graph node "
                "exactly once."
            )
        for node_tag in graph_section.find_all(attrs={"data-graph-node": True}):
            node_id = str(node_tag.get("data-graph-node") or "")
            hit_target = node_tag.find(attrs={"data-graph-node-hit": node_id})
            if (
                hit_target is None
                or hit_target.name != "circle"
                or str(node_tag.get("role") or "") != "button"
                or node_tag.get("tabindex") is None
            ):
                raise ValueError(
                    "Kimi Code report Molecular Search Graph node lacks its "
                    f"accessible enlarged pointer target: {node_id}"
                )
            if str(node_tag.get("data-molecule-number") or "") != str(
                molecule_number_by_id[node_id]
            ):
                raise ValueError(
                    "Kimi Code report Molecular Search Graph node has a missing or "
                    f"incorrect molecule number: {node_id}"
                )
        if (
            len(graph_detail_ids) != len(expected_graph_node_ids)
            or len(set(graph_detail_ids)) != len(graph_detail_ids)
            or set(graph_detail_ids) != expected_graph_node_ids
        ):
            raise ValueError(
                "Kimi Code report Molecular Search Graph must provide one selectable "
                "detail panel for every graph node."
            )
        for detail in graph_section.find_all(attrs={"data-graph-node-detail": True}):
            node_id = str(detail.get("data-graph-node-detail") or "")
            if (
                str(detail.get("data-molecule-number") or "")
                != str(molecule_number_by_id[node_id])
                or molecule_label(node_id)
                not in html.unescape(detail.get_text(" ", strip=True))
            ):
                raise ValueError(
                    "Kimi Code report graph detail has a missing or incorrect "
                    f"human-facing molecule label: {node_id}"
                )
        if expected_graph_edges is not None:
            expected_edge_keys = sorted(
                f"{str(edge.get('source') or '')}|{str(edge.get('target') or '')}"
                for edge in expected_graph_edges
                if str(edge.get("source") or "") in expected_graph_node_ids
                and str(edge.get("target") or "") in expected_graph_node_ids
            )
            actual_edge_keys = sorted(
                str(tag.get("data-graph-edge") or "")
                for tag in graph_section.find_all(attrs={"data-graph-edge": True})
            )
            if actual_edge_keys != expected_edge_keys:
                raise ValueError(
                    "Kimi Code report Molecular Search Graph must render every valid "
                    "search-graph edge exactly once."
                )

        for link in soup.find_all("link"):
            if str(link.get("rel") or "").lower().find("stylesheet") >= 0:
                raise ValueError("Kimi Code report depends on an external stylesheet.")
        for tag in soup.find_all(src=True):
            src = str(tag.get("src") or "").strip().lower()
            if src.startswith(("http://", "https://", "//")):
                raise ValueError(f"Kimi Code report contains an external resource: {src[:120]}")

        rendered = html.unescape(soup.get_text("\n") + "\n" + html_text)
        missing_smiles = [
            str(node.get("smiles") or "").strip()
            for node in candidate_nodes
            if str(node.get("smiles") or "").strip()
            and str(node.get("smiles") or "").strip() not in rendered
        ]
        if missing_smiles:
            raise ValueError(
                "Kimi Code report omitted candidate SMILES from the complete graph audit: "
                + ", ".join(missing_smiles[:5])
            )

        candidate_by_id = {
            str(node.get("id") or ""): node for node in candidate_nodes
        }
        candidate_ids = set(candidate_by_id)
        if re.search(r"text-overflow\s*:\s*ellipsis|-webkit-line-clamp", html_text, re.I):
            raise ValueError(
                "Kimi Code report visually truncates content; full SMILES must wrap "
                "without ellipsis or line clamping."
            )
        detailed_cards = soup.find_all(attrs={"data-candidate-card": True})
        if expected_detailed_cards is not None and len(detailed_cards) != expected_detailed_cards:
            raise ValueError(
                "Kimi Code report rendered the wrong number of detailed candidate cards: "
                f"expected {expected_detailed_cards}, got {len(detailed_cards)}."
            )
        detailed_card_ids: list[str] = []
        for card in detailed_cards:
            card_id = str(card.get("data-candidate-card") or "")
            if card_id not in candidate_ids:
                raise ValueError(
                    f"Kimi Code report contains a card for an unknown candidate: {card_id}"
                )
            if card_id in detailed_card_ids:
                raise ValueError(
                    f"Kimi Code report repeats a detailed candidate card: {card_id}"
                )
            card_visible_text = html.unescape(card.get_text(" ", strip=True))
            if (
                str(card.get("data-molecule-number") or "")
                != str(molecule_number_by_id[card_id])
                or molecule_label(card_id) not in card_visible_text
            ):
                raise ValueError(
                    "Kimi Code report candidate card lacks its stable molecule number: "
                    f"{card_id}"
                )
            comparison = card.find(attrs={"data-comparison-node": card_id})
            if comparison is None:
                raise ValueError(
                    "Kimi Code report candidate card lacks its matching parent comparison: "
                    f"{card_id}"
                )
            card_smiles = str(candidate_by_id[card_id].get("smiles") or "").strip()
            if not card_smiles or card_smiles not in card_visible_text:
                raise ValueError(
                    "Kimi Code report candidate card does not visibly label the molecule "
                    f"with its full SMILES: {card_id}"
                )
            full_smiles_label = card.find(attrs={"data-full-smiles": card_id})
            if (
                full_smiles_label is None
                or html.unescape(full_smiles_label.get_text("", strip=True)) != card_smiles
            ):
                raise ValueError(
                    "Kimi Code report candidate card lacks its marked, complete SMILES "
                    f"label: {card_id}"
                )
            property_grid = card.find(attrs={"data-property-grid": card_id})
            if property_grid is None or property_grid.find("table") is not None:
                raise ValueError(
                    "Kimi Code report candidate descriptors must use a wrapping property "
                    f"grid rather than a wide table: {card_id}"
                )
            if property_grid.get("data-property-columns") != "2":
                raise ValueError(
                    "Kimi Code report candidate descriptor grid must declare its "
                    f"two-column wrapping layout: {card_id}"
                )
            detailed_card_ids.append(card_id)

        candidate_grid = soup.find(attrs={"data-candidate-grid": True})
        if (
            candidate_grid is None
            or candidate_grid.get("data-desktop-columns") != "3"
            or any(
                card.find_parent(attrs={"data-candidate-grid": True}) is not candidate_grid
                for card in detailed_cards
            )
        ):
            raise ValueError(
                "Kimi Code report must place every focused card in one declared "
                "three-column desktop grid."
            )

        recommendation = soup.find(
            attrs={"data-recommendation-layout": "stacked-full-width"}
        )
        tier_grid = (
            recommendation.find(attrs={"data-tier-summary-grid": True})
            if recommendation is not None
            else None
        )
        tier_panels = (
            tier_grid.find_all(attrs={"data-recommendation-tier": True}, recursive=False)
            if tier_grid is not None
            else []
        )
        if (
            tier_grid is None
            or tier_grid.get("data-desktop-columns") != "3"
            or {str(panel.get("data-recommendation-tier") or "") for panel in tier_panels}
            != {"A", "B", "C"}
        ):
            raise ValueError(
                "Kimi Code report must render A/B/C recommendation tiers as one "
                "full-width three-column grid."
            )
        synthesis_table = (
            recommendation.find("table", attrs={"data-synthesis-shortlist": True})
            if recommendation is not None
            else None
        )
        if synthesis_table is None:
            raise ValueError(
                "Kimi Code report must render the synthesis shortlist as a full-width table."
            )
        synthesis_rows = synthesis_table.find_all(
            "tr", attrs={"data-synthesis-candidate": True}
        )
        synthesis_ids = [
            str(row.get("data-synthesis-candidate") or "") for row in synthesis_rows
        ]
        if expected_synthesis_count is not None and len(synthesis_ids) != expected_synthesis_count:
            raise ValueError(
                "Kimi Code report rendered the wrong number of synthesis-shortlist rows: "
                f"expected {expected_synthesis_count}, got {len(synthesis_ids)}."
            )
        if (
            len(synthesis_ids) != len(set(synthesis_ids))
            or any(candidate_id not in candidate_by_id for candidate_id in synthesis_ids)
        ):
            raise ValueError(
                "Kimi Code report synthesis shortlist contains duplicate or unknown candidates."
            )
        for row, candidate_id in zip(synthesis_rows, synthesis_ids):
            row_text = html.unescape(row.get_text(" ", strip=True))
            if (
                str(row.get("data-molecule-number") or "")
                != str(molecule_number_by_id[candidate_id])
                or molecule_label(candidate_id) not in row_text
            ):
                raise ValueError(
                    "Kimi Code report synthesis row omits the candidate's stable "
                    "molecule number: "
                    f"{candidate_id}"
                )

        audit_table = soup.find("table", attrs={"data-full-node-audit": True})
        if audit_table is None:
            raise ValueError(
                "Kimi Code report must provide the complete node audit as one compact table."
            )
        audit_rows = audit_table.find_all("tr", attrs={"data-audit": True})
        audit_row_ids = [str(row.get("data-audit") or "") for row in audit_rows]
        allowed_audit_ids = candidate_ids | set(root_node_ids)
        unknown_audit_ids = sorted(set(audit_row_ids) - allowed_audit_ids)
        if unknown_audit_ids:
            raise ValueError(
                "Kimi Code report full-node audit table contains unknown nodes: "
                + ", ".join(unknown_audit_ids[:5])
            )
        if (
            len(audit_row_ids) != len(allowed_audit_ids)
            or set(audit_row_ids) != allowed_audit_ids
        ):
            raise ValueError(
                "Kimi Code report Full-Node Audit must contain every graph molecule "
                "exactly once, including the root."
            )
        if len(set(audit_row_ids)) != len(audit_row_ids):
            raise ValueError(
                "Kimi Code report repeats a candidate in the full-node audit table."
            )
        audit_numbers = [molecule_number_by_id[node_id] for node_id in audit_row_ids]
        if audit_numbers != sorted(audit_numbers):
            raise ValueError(
                "Kimi Code report Full-Node Audit rows must follow molecule-number order."
            )
        for row, candidate_id in zip(audit_rows, audit_row_ids):
            topology = row.find(attrs={"data-audit-topology": candidate_id})
            if topology is None or topology.find("svg") is None:
                raise ValueError(
                    "Kimi Code report full-node audit row lacks its topology thumbnail: "
                    f"{candidate_id}"
                )
            row_text = html.unescape(row.get_text(" ", strip=True))
            if (
                str(row.get("data-molecule-number") or "")
                != str(molecule_number_by_id[candidate_id])
                or molecule_label(candidate_id) not in row_text
            ):
                raise ValueError(
                    "Kimi Code report Full-Node Audit row lacks its stable molecule "
                    f"number: {candidate_id}"
                )
            if candidate_id not in candidate_by_id:
                continue
            candidate = candidate_by_id[candidate_id]
            candidate_smiles = str(candidate.get("smiles") or "").strip()
            parent_smiles = [
                str(value).strip()
                for value in candidate.get("parent_smiles", [])
                if str(value).strip()
            ]
            if not candidate_smiles or candidate_smiles not in row_text:
                raise ValueError(
                    "Kimi Code report full-node audit row omits the candidate SMILES: "
                    f"{candidate_id}"
                )
            full_smiles_label = row.find(attrs={"data-full-smiles": candidate_id})
            if (
                full_smiles_label is None
                or html.unescape(full_smiles_label.get_text("", strip=True))
                != candidate_smiles
            ):
                raise ValueError(
                    "Kimi Code report full-node audit row lacks its marked, complete "
                    f"SMILES label: {candidate_id}"
                )
            if parent_smiles and not any(value in row_text for value in parent_smiles):
                raise ValueError(
                    "Kimi Code report full-node audit row omits its direct-parent SMILES: "
                    f"{candidate_id}"
                )
            known_parent_ids = [
                str(value)
                for value in candidate.get("parent_ids", [])
                if str(value) in molecule_number_by_id
            ]
            if known_parent_ids and not any(
                molecule_label(parent_id) in row_text
                for parent_id in known_parent_ids
            ):
                raise ValueError(
                    "Kimi Code report Full-Node Audit row omits its direct parent's "
                    f"molecule number: {candidate_id}"
                )
            if row.find(attrs={"data-comparison-node": True}) is not None:
                raise ValueError(
                    "Kimi Code report expanded molecule comparisons inside the compact "
                    f"full-node audit table: {candidate_id}"
                )
        diff_nodes = {
            str(tag.get("data-diff-node") or "")
            for tag in soup.find_all(attrs={"data-diff-node": True})
        }
        candidate_diff_nodes = diff_nodes & candidate_ids
        if not candidate_diff_nodes:
            raise ValueError(
                "Kimi Code report contains no parent-relative highlighted candidate depiction."
            )
        unknown_diff_nodes = sorted(diff_nodes - candidate_ids - set(root_node_ids))
        if unknown_diff_nodes:
            raise ValueError(
                "Kimi Code report labels unknown nodes as molecule differences: "
                + ", ".join(unknown_diff_nodes[:5])
            )
        parent_ids_by_child = {
            str(node.get("id") or ""): {
                str(parent_id)
                for parent_id in node.get("parent_ids", [])
                if str(parent_id)
            }
            for node in candidate_nodes
        }
        comparison_nodes: set[str] = set()
        for wrapper in soup.find_all(attrs={"data-comparison-node": True}):
            child_id = str(wrapper.get("data-comparison-node") or "")
            if child_id not in candidate_ids:
                raise ValueError(
                    f"Kimi Code report contains a comparison for unknown child node: {child_id}"
                )
            child_visual = wrapper.find(attrs={"data-diff-node": child_id})
            parent_visual = wrapper.find(attrs={"data-parent-node": True})
            if child_visual is None or parent_visual is None:
                raise ValueError(
                    f"Kimi Code report comparison is missing its parent or child visual: {child_id}"
                )
            parent_id = str(parent_visual.get("data-parent-node") or "")
            if parent_id not in parent_ids_by_child.get(child_id, set()):
                raise ValueError(
                    "Kimi Code report compares a child with a node that is not its direct parent: "
                    f"{parent_id} -> {child_id}"
                )
            comparison_text = html.unescape(wrapper.get_text(" ", strip=True))
            if (
                molecule_label(parent_id) not in comparison_text
                or molecule_label(child_id) not in comparison_text
            ):
                raise ValueError(
                    "Kimi Code report parent-child comparison must use molecule "
                    f"numbers as its primary labels: {parent_id} -> {child_id}"
                )
            comparison_nodes.add(child_id)
        missing_comparisons = sorted(candidate_diff_nodes - comparison_nodes)
        if missing_comparisons:
            raise ValueError(
                "Kimi Code report shows highlighted children without a parent comparison: "
                + ", ".join(missing_comparisons[:5])
            )
        for reference in soup.find_all(attrs={"data-molecule-ref": True}):
            node_id = str(reference.get("data-molecule-ref") or "")
            if node_id not in molecule_number_by_id:
                raise ValueError(
                    f"Kimi Code report contains an unknown molecule reference: {node_id}"
                )
            if (
                str(reference.get("data-molecule-number") or "")
                != str(molecule_number_by_id[node_id])
                or molecule_label(node_id)
                not in html.unescape(reference.get_text(" ", strip=True))
            ):
                raise ValueError(
                    "Kimi Code report contains an inconsistent molecule-number "
                    f"reference: {node_id}"
                )
        for risk_group in soup.find_all(attrs={"data-risk-theme": True}):
            risk_candidate_ids = [
                str(reference.get("data-molecule-ref") or "")
                for reference in risk_group.find_all(
                    attrs={"data-molecule-ref": True}
                )
                if str(reference.get("data-molecule-ref") or "") in candidate_ids
            ]
            repeated_ids = sorted(
                candidate_id
                for candidate_id in set(risk_candidate_ids)
                if risk_candidate_ids.count(candidate_id) > 1
            )
            if repeated_ids:
                theme = str(risk_group.get("data-risk-theme") or "unknown")
                raise ValueError(
                    "Kimi Code report repeats candidates within one grouped risk theme "
                    f"({theme}): " + ", ".join(repeated_ids[:5])
                )
        visible_soup = BeautifulSoup(html_text, "html.parser")
        for tag in visible_soup.find_all(("script", "style")):
            tag.decompose()
        visible_text = html.unescape(visible_soup.get_text(" ", strip=True))
        visible_node_ids = [
            node_id for node_id in expected_graph_node_ids if node_id in visible_text
        ]
        if visible_node_ids:
            raise ValueError(
                "Kimi Code report exposes internal node IDs as visible molecule labels: "
                + ", ".join(sorted(visible_node_ids)[:5])
            )
        saved_state = soup.find(id="saved-review-state")
        if saved_state is None or saved_state.get("data-state") is None:
            raise ValueError(
                "Kimi Code report does not provide attribute-based persistent review state."
            )
        manual_state_escaping = re.search(
            r"setAttribute\s*\(\s*['\"]data-state['\"]\s*,\s*"
            r"JSON\.stringify\([^)]*\)\s*\.replace",
            html_text,
        )
        if manual_state_escaping:
            raise ValueError(
                "Kimi Code report manually HTML-escapes JSON before setAttribute; "
                "this breaks review-state restoration after HTML export."
            )
        if str(PROJECT_ROOT) in html_text:
            raise ValueError("Kimi Code report leaked the private M3OS project path.")

    @staticmethod
    def _positive_int_env(name: str, default: int) -> int:
        value = int(os.getenv(name) or default)
        if value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return value

    @staticmethod
    def _positive_float_env(name: str, default: float) -> float:
        value = float(os.getenv(name) or default)
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")
        return value

    @staticmethod
    def _nonnegative_int_env(name: str, default: int) -> int:
        raw_value = os.getenv(name)
        value = int(raw_value) if raw_value not in (None, "") else int(default)
        if value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
        return value

    @staticmethod
    def _nonnegative_float_env(name: str, default: float) -> float:
        raw_value = os.getenv(name)
        value = float(raw_value) if raw_value not in (None, "") else float(default)
        if value < 0:
            raise ValueError(f"{name} must be greater than or equal to zero.")
        return value

    @staticmethod
    def _tail(text: str, max_chars: int = 4000) -> str:
        cleaned = str(text or "").strip()
        return cleaned[-max_chars:]
