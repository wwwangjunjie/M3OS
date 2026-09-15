"""Multi-turn main agent/session controller for agents_v5.

The main agent owns user interaction and orchestration. It prepares tasks,
initializes and selects MCGS nodes, builds shared analysis, runs the narrow
expansion workflow, and decides when to stop/export. The molecular generation
and evaluation core remains in the creative/rational/critic expansion workflow.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import StructuredTool
from langchain.tools import tool

from m3os.agents_v5.agents.critic import (
    CriticAgent,
    create_critic_agent,
)
from m3os.agents_v5.agents.auditor import (
    AuditorAgent,
    create_auditor_agent,
)
from m3os.agents_v5.agents.creative_explorer import (
    CreativeMoleculeExplorerAgent,
    create_creative_molecule_explorer_agent,
)
from m3os.agents_v5.agents.rational_designer import (
    RationalMedicinalDesignerAgent,
    create_rational_medicinal_designer_agent,
)
from m3os.agents_v5.agents.medchem_retrieval import (
    MedChemRetrievalAgent,
    create_medchem_retrieval_agent,
)
from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.core.retry_utils import invoke_with_retry
from m3os.agents_v5.core.state import MCGSState
from m3os.agents_v5.core.trace_utils import structured_field, structured_optimizations
from m3os.agents_v5.prompts.system_loader import render_main_system_prompt
from m3os.agents_v5.services.document_rag import DocumentIngestionResult, DocumentRAGService
from m3os.agents_v5.services.knowledge_context import KnowledgeContextService
from m3os.agents_v5.services.llm_factory import create_base_model, get_middleware
from m3os.agents_v5.services.mcp_client import MCPClientManager, TOOL_NAMES_MCP
from m3os.agents_v5.services.kimi_code_report import KimiCodeReportGenerator
from m3os.agents_v5.services.smiles_validation import (
    compact_invalid_molecules,
    split_valid_molecules,
    tanimoto_similarity_to_root,
    validate_smiles,
)
from m3os.agents_v5.services.task_preparation import TaskPreparationService
from m3os.agents_v5.services.table_context import (
    build_table_context,
    combine_context_blocks,
)
from m3os.agents_v5.services.structure_files import build_structure_file_context
from m3os.agents_v5.tools.monte_carlo_graph.graph_for_molecular_optimization_v3_4 import (
    MoleculeEdge,
    MoleculeNode,
)
from m3os.agents_v5.workflow import (
    AnalyzeCurrentNode,
    InitGraphNode,
    SelectBestCandidateNode,
    create_mcgs_expansion_workflow,
)


MAIN_AGENT_SYSTEM_PROMPT = render_main_system_prompt()

AgentEventSink = Callable[[Dict[str, Any]], Awaitable[None] | None]
MIN_LANGGRAPH_RECURSION_LIMIT = 100
MAIN_AGENT_EXCLUDED_TOOL_NAMES = {"write_todos"}


def _merge_upload_file_paths(*path_groups: Optional[List[str]]) -> List[str]:
    """Merge optional upload path groups while preserving order."""
    merged: List[str] = []
    seen: set[str] = set()
    for path_group in path_groups:
        for item in path_group or []:
            path_text = str(item).strip()
            if not path_text or path_text in seen:
                continue
            merged.append(path_text)
            seen.add(path_text)
    return merged


def _build_uploaded_documents_full_text_context(
    document_result: Optional[DocumentIngestionResult],
) -> Optional[str]:
    """Render converted uploaded-document Markdown as a reusable context block."""
    if not document_result or not document_result.markdown_paths:
        return None

    source_files = list(document_result.metadata.get("source_files", []))
    sections: List[str] = []
    loaded_names: List[str] = []
    for index, markdown_file in enumerate(document_result.markdown_paths):
        path = Path(markdown_file)
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        source_file = source_files[index] if index < len(source_files) else None
        source_name = Path(source_file).name if source_file else path.name
        loaded_names.append(source_name)
        sections.append(
            "\n".join(
                [
                    f"## Document: {source_name}",
                    f"Converted Markdown file: {path.name}",
                    "",
                    text,
                ]
            )
        )

    if not sections:
        return None

    names_text = ", ".join(loaded_names)
    return (
        "[UPLOADED DOCUMENT FULL TEXT - CONVERTED MARKDOWN]\n"
        "The following uploaded documents were converted to Markdown before this "
        "agent session started. Treat this block as the authoritative uploaded-document "
        "context for the current session. Use it directly; vector RAG and chunk retrieval "
        "are disabled for uploaded documents in this mode.\n"
        f"Covered documents: {names_text}\n\n"
        + "\n\n---\n\n".join(sections)
        + "\n[END UPLOADED DOCUMENT FULL TEXT]\n"
    )


def _build_uploaded_documents_summary_context(
    document_result: Optional[DocumentIngestionResult] = None,
    document_info: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Render uploaded-document metadata without converted full text."""
    info = document_info or {}
    if document_result is not None:
        metadata = document_result.metadata or {}
        source_files = list(metadata.get("source_files", []))
        markdown_files = list(metadata.get("markdown_files", document_result.markdown_paths))
        topics = list(metadata.get("topics", []))
        documents = []
        for index, markdown_file in enumerate(markdown_files):
            source_file = source_files[index] if index < len(source_files) else None
            source_name = Path(source_file).name if source_file else Path(markdown_file).name
            documents.append({"source_name": source_name})
        info = {
            "kb_id": document_result.kb_id,
            "documents": documents,
            "document_count": len(documents),
            "topics": topics,
            "total_chars": sum(_safe_file_chars(path) for path in markdown_files),
        }
    documents = info.get("documents") or []
    if not documents:
        return None
    names = [str(item.get("source_name") or item.get("markdown_name") or "").strip() for item in documents]
    names_text = "\n".join(f"- {name}" for name in names if name)
    topics = [str(item) for item in (info.get("topics") or [])[:12] if str(item).strip()]
    topics_text = "\n".join(f"- {topic}" for topic in topics)
    parts = [
        "[UPLOADED DOCUMENT SOURCE SUMMARY]",
        f"Uploaded document count: {info.get('document_count') or len(documents)}",
        f"Converted Markdown context id: {info.get('kb_id') or 'N/A'}",
        f"Total converted Markdown characters: {info.get('total_chars', 0)}",
        "Documents:",
        names_text or "- N/A",
    ]
    if topics_text:
        parts.extend(["Detected document headings/topics:", topics_text])
    parts.append(
        "Full uploaded-document text is available to the dedicated medicinal-chemistry "
        "retrieval agent. Ask `ask_medchem_knowledge` when these documents may affect "
        "SAR, ADMET, mechanism, scaffold risk, or candidate scoring."
    )
    return "\n".join(parts)


def _safe_file_chars(path_text: Any) -> int:
    try:
        path = Path(str(path_text))
        if not path.exists() or not path.is_file():
            return 0
        return len(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return 0


def _build_uploaded_table_summary_context(table_context: Optional[str]) -> Optional[str]:
    """Render table metadata without full table rows."""
    if not table_context:
        return None
    selected_prefixes = (
        "## Table Source",
        "Path:",
        "Format:",
        "Sheet:",
        "Rows:",
        "Columns:",
        "Column names:",
    )
    lines = []
    for raw_line in str(table_context).splitlines():
        line = raw_line.strip()
        if any(line.startswith(prefix) for prefix in selected_prefixes):
            lines.append(line)
    if not lines:
        return None
    return (
        "[UPLOADED TABLE SOURCE SUMMARY]\n"
        "Uploaded CSV/Excel tables are available in full to Main and the dedicated "
        "medicinal-chemistry retrieval agent. Specialist agents see only this source "
        "summary.\n"
        + "\n".join(lines)
        + "\nAsk `ask_medchem_knowledge` when table-supported SAR, assay, ADMET, PK, "
        "or measured-value trends may affect a design or scoring decision."
    )


def _build_specialist_upload_awareness_context(
    *,
    table_context: Optional[str] = None,
    document_result: Optional[DocumentIngestionResult] = None,
    document_info: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    return combine_context_blocks(
        _build_uploaded_table_summary_context(table_context),
        _build_uploaded_documents_summary_context(document_result, document_info),
    )


class MainAgentToolExclusionMiddleware(AgentMiddleware):
    """Hide selected DeepAgents built-in tools from the main controller model."""

    def __init__(self, excluded_tool_names: set[str]):
        self.excluded_tool_names = {str(name).strip() for name in excluded_tool_names if str(name).strip()}

    @staticmethod
    def _tool_name(tool_item: Any) -> str:
        if isinstance(tool_item, dict):
            name = tool_item.get("name")
        else:
            name = getattr(tool_item, "name", None)
        return str(name or "")

    def _filter_request(self, request: Any) -> Any:
        if not self.excluded_tool_names:
            return request
        filtered_tools = [
            tool_item
            for tool_item in request.tools
            if self._tool_name(tool_item) not in self.excluded_tool_names
        ]
        return request.override(tools=filtered_tools)

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(self._filter_request(request))

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        return await handler(self._filter_request(request))


class M3OSChatSession:
    """In-memory multi-turn session for the agents_v5 molecular optimizer."""

    def __init__(
        self,
        config: AgentConfig,
        mcp_client: MCPClientManager,
        creative_agent: CreativeMoleculeExplorerAgent,
        rational_agent: RationalMedicinalDesignerAgent,
        critic_agent: CriticAgent,
        medchem_retrieval_agent: Optional[MedChemRetrievalAgent] = None,
        auditor_agent: Optional[AuditorAgent] = None,
        knowledge_context: Optional[KnowledgeContextService] = None,
        document_kb_id: Optional[str] = None,
        document_kb_metadata: Optional[Dict[str, Any]] = None,
        additional_context: Optional[str] = None,
        main_additional_context: Optional[str] = None,
        table_context: Optional[str] = None,
        structure_context: Optional[str] = None,
        protein_fasta_path: Optional[str] = None,
        leadopt_task_contract: Optional[Dict[str, Any]] = None,
        uploaded_document_context: Optional[str] = None,
        recursion_limit: int = 400,
        event_sink: Optional[AgentEventSink] = None,
        enabled_generators: Optional[List[str]] = None,
    ):
        self.config = config
        self.mcp_client = mcp_client
        self.creative_agent = creative_agent
        self.rational_agent = rational_agent
        self.critic_agent = critic_agent
        self.medchem_retrieval_agent = medchem_retrieval_agent
        self.auditor_agent = auditor_agent or AuditorAgent(config)
        self.document_kb_id = document_kb_id
        self.document_kb_metadata = document_kb_metadata or {}
        self.knowledge_context = knowledge_context or KnowledgeContextService(
            config,
            document_kb_id=document_kb_id,
            document_kb_metadata=self.document_kb_metadata,
            mcp_client=mcp_client,
        )
        if knowledge_context is not None:
            set_mcp_client = getattr(self.knowledge_context, "set_mcp_client", None)
            if callable(set_mcp_client):
                set_mcp_client(mcp_client)
        self.additional_context = additional_context
        self.main_additional_context = main_additional_context
        self.table_context = table_context
        self.structure_context = structure_context
        self.protein_fasta_path = protein_fasta_path
        self.leadopt_task_contract = dict(leadopt_task_contract or {})
        self.uploaded_document_context = uploaded_document_context
        self.recursion_limit = self._normalize_recursion_limit(recursion_limit)
        self.enabled_generators = self._normalize_enabled_generators(enabled_generators)
        self.event_sink: Optional[AgentEventSink] = None
        self._set_event_sink(event_sink)
        self._set_medchem_retrieval_agent(medchem_retrieval_agent)

        self.messages: List[Dict[str, str]] = []
        self.state: Optional[MCGSState] = None
        self.last_search_result: Optional[MCGSState] = None
        self.tool_cache: Dict[str, Any] = self._new_tool_cache()
        self._main_agent: Optional[Any] = None
        self._main_agent_tools: Optional[List[Any]] = None
        self._cancel_event = asyncio.Event()
        self._cancel_reason: Optional[str] = None
        self._current_run_task: Optional[asyncio.Task[Any]] = None

        self.task_preparation = TaskPreparationService(
            config=config,
            mcp_client=mcp_client,
            creative_agent=creative_agent,
            rational_agent=rational_agent,
            critic_agent=critic_agent,
            knowledge_context=self.knowledge_context,
        )
        self.init_graph_node = InitGraphNode(config, mcp_client, critic_agent, self.auditor_agent)
        self.select_candidate_node = SelectBestCandidateNode(config, mcp_client)
        self.analyze_current_node = AnalyzeCurrentNode(config)
        self.expansion_workflow = create_mcgs_expansion_workflow(
            config=config,
            creative_agent=creative_agent,
            rational_agent=rational_agent,
            critic_agent=critic_agent,
            auditor_agent=self.auditor_agent,
            enabled_generators=self.enabled_generators,
        )
        # Report generation is an isolated Kimi Code coding task and does not
        # use the legacy LangChain ReportAgent/blueprint pipeline.
        self.report_generator = KimiCodeReportGenerator(config=config)

    @classmethod
    async def create(
        cls,
        env_file: Optional[str] = None,
        additional_context: Optional[str] = None,
        table_context_paths: Optional[List[str]] = None,
        structure_file_paths: Optional[List[str]] = None,
        protein_fasta_path: Optional[str] = None,
        leadopt_task_contract: Optional[Dict[str, Any]] = None,
        document_files: Optional[List[str]] = None,
        document_dir: Optional[str] = None,
        recursion_limit: int = 400,
        event_sink: Optional[AgentEventSink] = None,
        enabled_generators: Optional[List[str]] = None,
        enable_medchem_retrieval: bool = True,
        enable_molecular_auxiliary_context: bool = True,
        enable_optimization_case_retrieval: bool = True,
        kimi_reasoning_effort: Optional[str] = None,
    ) -> "M3OSChatSession":
        """Create a fully initialized chat session."""
        config = AgentConfig.from_env_file(env_file) if env_file else AgentConfig.from_env()
        if not enable_optimization_case_retrieval:
            disabled_tool_names = {
                name
                for name in os.getenv(
                    "M3OS_DISABLED_MCP_TOOL_NAMES", ""
                ).replace(",", " ").split()
                if name
            }
            disabled_tool_names.add("query_evolutionary_optimization_cases")
            os.environ["M3OS_DISABLED_MCP_TOOL_NAMES"] = ",".join(
                sorted(disabled_tool_names)
            )
        if kimi_reasoning_effort is not None:
            normalized_effort = str(kimi_reasoning_effort).strip().lower()
            if normalized_effort not in {"low", "high", "max"}:
                raise ValueError(
                    "kimi_reasoning_effort must be one of: low, high, max"
                )
            config.llm.kimi_reasoning_effort = normalized_effort
        config.enable_molecular_auxiliary_context = enable_molecular_auxiliary_context
        config.deepagent.enable_optimization_case_retrieval = (
            enable_optimization_case_retrieval
        )
        task_contract_context = None
        if leadopt_task_contract:
            task_contract_context = (
                "[STRUCTURED LEAD-OPTIMIZATION TASK CONTRACT]\n"
                f"{json.dumps(leadopt_task_contract, ensure_ascii=False, sort_keys=True)}\n"
                f"protein_fasta_path: {protein_fasta_path or ''}\n"
                "[END STRUCTURED LEAD-OPTIMIZATION TASK CONTRACT]"
            )
        base_additional_context = combine_context_blocks(
            additional_context,
            task_contract_context,
        )
        table_context = None
        try:
            table_context = await build_table_context(table_context_paths)
        except Exception as exc:
            print(f"Failed to build table context; continuing with existing context: {exc}")
        main_additional_context = combine_context_blocks(base_additional_context)
        structure_context = None
        try:
            structure_context = build_structure_file_context(
                _merge_upload_file_paths(
                    structure_file_paths,
                    table_context_paths,
                    document_files,
                ),
                directory=document_dir,
                protein_fasta_path=protein_fasta_path,
            )
        except Exception as exc:
            print(f"Failed to process uploaded SDF/PDB files; continuing with existing context: {exc}")
        additional_context = combine_context_blocks(base_additional_context, table_context, structure_context)
        if table_context:
            print(f"Table context assembled: {len(table_context)} characters")
        if structure_context:
            print(f"SDF/PDB structure context assembled: {len(structure_context)} characters")

        document_result = None
        uploaded_document_context = None
        document_service = DocumentRAGService(config.paths)
        supported_document_files = document_service.collect_document_files(document_files, document_dir)
        if supported_document_files:
            document_result = document_service.prepare_documents_full_text_fail_fast(
                document_files=supported_document_files,
            )
            uploaded_document_context = _build_uploaded_documents_full_text_context(document_result)
            additional_context = combine_context_blocks(additional_context, uploaded_document_context)
            print(
                "Preprocessed user documents for full-text system prompt: "
                f"source_files={len(document_result.metadata.get('source_files', []))}, "
                f"kb_id={document_result.kb_id}, "
                f"markdown_files={len(document_result.markdown_paths)}"
            )

        mcp_client = MCPClientManager(config.mcp)
        await mcp_client.initialize()
        try:
            specialist_upload_awareness = _build_specialist_upload_awareness_context(
                table_context=table_context,
                document_result=document_result,
            )
            specialist_additional_context = combine_context_blocks(
                base_additional_context,
                structure_context,
                specialist_upload_awareness,
            )
            medchem_retrieval_context = combine_context_blocks(
                base_additional_context,
                table_context,
                uploaded_document_context,
            )

            medchem_retrieval_agent = None
            if enable_medchem_retrieval:
                medchem_retrieval_agent = await create_medchem_retrieval_agent(
                    config,
                    mcp_client,
                    additional_context=medchem_retrieval_context,
                )

            creative_agent = await create_creative_molecule_explorer_agent(
                config,
                mcp_client,
                additional_context=specialist_additional_context,
                medchem_retrieval_agent=medchem_retrieval_agent,
            )
            rational_agent = await create_rational_medicinal_designer_agent(
                config,
                mcp_client,
                additional_context=specialist_additional_context,
                medchem_retrieval_agent=medchem_retrieval_agent,
            )
            critic_agent = await create_critic_agent(
                config,
                mcp_client,
                additional_context=specialist_additional_context,
                medchem_retrieval_agent=medchem_retrieval_agent,
            )
            auditor_agent = await create_auditor_agent(config)
            document_kb_id = document_result.kb_id if document_result else None
            document_kb_metadata = document_result.metadata if document_result else {}
            knowledge_context = KnowledgeContextService(
                config,
                document_kb_id=document_kb_id,
                document_kb_metadata=document_kb_metadata,
                mcp_client=mcp_client,
            )

            return cls(
                config=config,
                mcp_client=mcp_client,
                creative_agent=creative_agent,
                rational_agent=rational_agent,
                critic_agent=critic_agent,
                medchem_retrieval_agent=medchem_retrieval_agent,
                auditor_agent=auditor_agent,
                knowledge_context=knowledge_context,
                document_kb_id=document_kb_id,
                document_kb_metadata=document_kb_metadata,
                additional_context=specialist_additional_context,
                main_additional_context=main_additional_context,
                table_context=table_context,
                structure_context=structure_context,
                protein_fasta_path=protein_fasta_path,
                leadopt_task_contract=leadopt_task_contract,
                uploaded_document_context=uploaded_document_context,
                recursion_limit=recursion_limit,
                event_sink=event_sink,
                enabled_generators=enabled_generators,
            )
        except BaseException:
            # Session construction can fail after stdio MCP subprocesses have
            # started (for example, when LLM configuration is invalid). Close
            # them before re-raising so a benchmark can continue to the next
            # record without accumulating orphaned MCP processes.
            try:
                await mcp_client.close()
            except Exception:
                # Do not mask the construction failure with cleanup noise.
                pass
            raise

    async def add_uploaded_contexts(
        self,
        *,
        additional_context: Optional[str] = None,
        table_context_paths: Optional[List[str]] = None,
        structure_file_paths: Optional[List[str]] = None,
        document_files: Optional[List[str]] = None,
        document_dir: Optional[str] = None,
    ) -> bool:
        """Merge newly uploaded table/document context into an existing session."""
        changed = False

        new_table_context = None
        if table_context_paths:
            try:
                new_table_context = await build_table_context(table_context_paths)
            except Exception as exc:
                print(f"Failed to add table context; continuing with existing context: {exc}")
            if new_table_context:
                self.table_context = combine_context_blocks(self.table_context, new_table_context)
                changed = True

        new_structure_context = None
        try:
            new_structure_context = build_structure_file_context(
                _merge_upload_file_paths(
                    structure_file_paths,
                    table_context_paths,
                    document_files,
                ),
                directory=document_dir,
                protein_fasta_path=self.protein_fasta_path,
            )
        except Exception as exc:
            print(f"Failed to process additional SDF/PDB files; continuing with existing context: {exc}")
        if new_structure_context:
            self.structure_context = combine_context_blocks(
                self.structure_context,
                new_structure_context,
            )
            changed = True

        new_document_context = None
        document_service = DocumentRAGService(self.config.paths)
        supported_document_files = document_service.collect_document_files(document_files, document_dir)
        if supported_document_files:
            document_result = document_service.prepare_documents_full_text_fail_fast(
                document_files=supported_document_files,
            )
            new_document_context = _build_uploaded_documents_full_text_context(document_result)
            if new_document_context:
                self.uploaded_document_context = combine_context_blocks(
                    self.uploaded_document_context,
                    new_document_context,
                )
                changed = True
            self.document_kb_id = document_result.kb_id
            self.document_kb_metadata = document_result.metadata
            self.knowledge_context.document_kb_id = document_result.kb_id
            self.knowledge_context.document_kb_metadata = document_result.metadata

        if additional_context and additional_context.strip():
            self.main_additional_context = combine_context_blocks(
                self.main_additional_context,
                additional_context,
            )
            changed = True

        if changed:
            document_info = self.knowledge_context.get_uploaded_file_info()
            specialist_upload_awareness = _build_specialist_upload_awareness_context(
                table_context=self.table_context,
                document_info=document_info,
            )
            self.additional_context = combine_context_blocks(
                self.main_additional_context,
                self.structure_context,
                specialist_upload_awareness,
            )
            medchem_retrieval_context = combine_context_blocks(
                self.main_additional_context,
                self.table_context,
                self.uploaded_document_context,
            )
            for agent in (
                self.creative_agent,
                self.rational_agent,
                self.critic_agent,
            ):
                update_context = getattr(agent, "update_additional_context", None)
                if callable(update_context):
                    update_context(self.additional_context or "")
            update_medchem_context = getattr(
                self.medchem_retrieval_agent,
                "update_additional_context",
                None,
            )
            if callable(update_medchem_context):
                update_medchem_context(medchem_retrieval_context or "")
            self._main_agent = None
        return changed

    async def close(self) -> None:
        """Close underlying MCP resources."""
        self.cancel_current_run("M3OS session is closing.")
        await self.mcp_client.close()

    def cancel_current_run(self, reason: str = "M3OS run stopped by user.") -> bool:
        """Request cancellation of the active turn, if any."""
        self._cancel_reason = reason
        self._cancel_event.set()
        task = self._current_run_task
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    def _reset_cancellation(self) -> None:
        self._cancel_reason = None
        self._cancel_event.clear()

    def _raise_if_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise asyncio.CancelledError(self._cancel_reason or "M3OS run cancelled.")

    def _set_event_sink(self, event_sink: Optional[AgentEventSink]) -> None:
        self.event_sink = event_sink
        for agent in (
            self.creative_agent,
            self.rational_agent,
            self.critic_agent,
            self.medchem_retrieval_agent,
        ):
            setter = getattr(agent, "set_event_sink", None)
            if callable(setter):
                setter(event_sink)

    def _set_medchem_retrieval_agent(
        self,
        medchem_retrieval_agent: Optional[MedChemRetrievalAgent],
    ) -> None:
        if medchem_retrieval_agent is None:
            return
        for agent in (self.creative_agent, self.rational_agent, self.critic_agent):
            setter = getattr(agent, "set_medchem_retrieval_agent", None)
            if callable(setter):
                setter(medchem_retrieval_agent)

    @staticmethod
    def _normalize_enabled_generators(enabled_generators: Optional[List[str]]) -> List[str]:
        if not enabled_generators:
            return ["creative", "rational"]
        aliases = {
            "creative_molecule_explorer": "creative",
            "creative_explorer": "creative",
            "rational_medicinal_designer": "rational",
            "rational_designer": "rational",
        }
        normalized = []
        for item in enabled_generators:
            name = aliases.get(str(item).strip().lower(), str(item).strip().lower())
            if name in {"creative", "rational"} and name not in normalized:
                normalized.append(name)
        return normalized or ["creative", "rational"]

    def set_enabled_generators(self, enabled_generators: Optional[List[str]]) -> List[str]:
        """Rebuild the expansion workflow with the selected generator agents."""
        self.enabled_generators = self._normalize_enabled_generators(enabled_generators)
        self.expansion_workflow = create_mcgs_expansion_workflow(
            config=self.config,
            creative_agent=self.creative_agent,
            rational_agent=self.rational_agent,
            critic_agent=self.critic_agent,
            auditor_agent=self.auditor_agent,
            enabled_generators=self.enabled_generators,
        )
        return list(self.enabled_generators)

    def set_generator_mode(self, mode: Optional[str]) -> List[str]:
        text = str(mode or "both").strip().lower()
        if text in {"both", "all", "default"}:
            return self.set_enabled_generators(["creative", "rational"])
        if text in {"rational", "rational_only", "rational-only"}:
            return self.set_enabled_generators(["rational"])
        if text in {"creative", "creative_only", "creative-only"}:
            return self.set_enabled_generators(["creative"])
        return self.set_enabled_generators([text])

    def load_conversation_history(self, history: Any) -> int:
        """Load persisted chat turns into a fresh session without duplicating the active turn."""
        if self.messages:
            return 0
        if not isinstance(history, list):
            return 0
        loaded: List[Dict[str, str]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            loaded.append({"role": role, "content": content[:12000]})
            if len(loaded) >= 40:
                break
        self.messages.extend(loaded)
        return len(loaded)

    async def ask(self, user_message: str) -> str:
        """Handle one user turn and return a user-facing response."""
        self._raise_if_cancelled()
        self.messages.append({"role": "user", "content": user_message})
        agent = await self._build_main_agent()
        async def _invoke_main_agent() -> Dict[str, Any]:
            self._raise_if_cancelled()
            return await agent.ainvoke(
                {"messages": self._messages_with_medchem_memory(self.messages)},
                config=self._langgraph_config(),
            )

        result = await invoke_with_retry(
            _invoke_main_agent,
            max_retries=3,
            base_delay=2.0,
            max_delay=2.0,
            jitter_ratio=0.0,
            validate_fn=self._validate_main_agent_result,
            operation_name="Main agent conversation",
        )
        self._raise_if_cancelled()
        response = self._extract_agent_response(result)
        self.messages.append({"role": "assistant", "content": response})
        return response

    async def ask_stream(self, user_message: str):
        """Yield frontend-friendly events for one user turn.

        DeepAgents streaming support varies by backend/version. This method tries
        to capture visible main-agent text before/after tool calls while keeping
        the existing event-sink path for tools, MCGS updates, and graph snapshots.
        If native streaming is unavailable before any side effects run, it falls
        back to the non-streaming conversation path.
        """
        self._reset_cancellation()
        queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        previous_sink = self.event_sink
        text_buffer: List[str] = []
        streamed_text_parts: List[str] = []

        async def flush_agent_message(stage: str = "reasoning") -> None:
            text = self._compact_text("".join(text_buffer), 1800)
            text_buffer.clear()
            if not text:
                return
            await self._emit_agent_message(
                message=text,
                agent_type="main_agent",
                agent_name="Main Agent",
                stage=stage,
            )

        async def queue_sink(event: Dict[str, Any]) -> None:
            self._raise_if_cancelled()
            if previous_sink is not None:
                result = previous_sink(event)
                if hasattr(result, "__await__"):
                    await result
            await queue.put(event)

        async def runner() -> None:
            self._set_event_sink(queue_sink)
            try:
                await self._emit_event(
                    "assistant_update",
                    "M3OS",
                    "M3OS main agent is analyzing the request.",
                )
                try:
                    response = await self._ask_with_visible_stream(
                        user_message,
                        on_text=lambda text: (
                            text_buffer.append(text),
                            streamed_text_parts.append(text),
                        ),
                        flush_agent_message=flush_agent_message,
                    )
                except RuntimeError as exc:
                    if str(exc) != "main-agent-streaming-unavailable":
                        raise
                    response = await self.ask(user_message)
                self._raise_if_cancelled()
                await self._emit_event("assistant_message", "M3OS", response)
            except asyncio.CancelledError:
                await queue.put({
                    "type": "assistant_update",
                    "source": "M3OS",
                    "content": {
                        "status": "stopped",
                        "message": self._cancel_reason or "M3OS run stopped by user.",
                    },
                    "timestamp": datetime.now(UTC).isoformat(),
                })
                raise
            except Exception as exc:
                await self._emit_event(
                    "error",
                    "M3OS",
                    f"{type(exc).__name__}: {exc}",
                )
            finally:
                self._set_event_sink(previous_sink)
                await queue.put(None)

        task = asyncio.create_task(runner())
        self._current_run_task = task
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            if self._current_run_task is task:
                self._current_run_task = None

    async def _ask_with_visible_stream(
        self,
        user_message: str,
        *,
        on_text: Callable[[str], Any],
        flush_agent_message: Callable[[str], Awaitable[None]],
    ) -> str:
        """Run the main agent through LangGraph streaming when available."""
        try:
            agent = await self._build_main_agent()
        except Exception as exc:
            raise RuntimeError("main-agent-streaming-unavailable") from exc
        astream_events = getattr(agent, "astream_events", None)
        if not callable(astream_events):
            raise RuntimeError("main-agent-streaming-unavailable")

        self.messages.append({"role": "user", "content": user_message})
        final_result: Any = None
        captured_text_parts: List[str] = []
        saw_event = False
        saw_side_effect = False
        try:
            async for event in astream_events(
                {"messages": self._messages_with_medchem_memory(self.messages)},
                config=self._langgraph_config(),
                version="v2",
            ):
                saw_event = True
                self._raise_if_cancelled()
                if not isinstance(event, dict):
                    continue
                event_name = str(event.get("event") or "")
                data = event.get("data") if isinstance(event.get("data"), dict) else {}
                if event_name == "on_chat_model_stream":
                    chunk_text = self._stream_chunk_text(data.get("chunk"))
                    if chunk_text:
                        captured_text_parts.append(chunk_text)
                        on_text(chunk_text)
                    continue
                if event_name in {"on_tool_start", "on_tool_end"}:
                    saw_side_effect = True
                    continue
                if event_name in {"on_chain_end", "on_graph_end"} and data.get("output") is not None:
                    final_result = data.get("output")
        except Exception:
            if not saw_event and not saw_side_effect:
                self.messages.pop()
                raise RuntimeError("main-agent-streaming-unavailable")
            raise

        if not saw_event:
            self.messages.pop()
            raise RuntimeError("main-agent-streaming-unavailable")

        response = self._extract_agent_response(final_result) if final_result is not None else ""
        if not response.strip():
            response = self._compact_text("".join(captured_text_parts), 4000)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def _messages_with_medchem_memory(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        memory_block = self._medchem_memory_block()
        if not memory_block:
            return messages
        updated_messages = [dict(message) for message in messages]
        for index in range(len(updated_messages) - 1, -1, -1):
            message = updated_messages[index]
            if message.get("role") != "user":
                continue
            content = str(message.get("content") or "")
            if "[SHARED MEDICINAL CHEMISTRY KNOWLEDGE MEMORY]" in content:
                return updated_messages
            message["content"] = f"{content.rstrip()}\n\n{memory_block}"
            updated_messages[index] = message
            return updated_messages
        return messages

    def _medchem_memory_block(self) -> str:
        formatter = getattr(self.mcp_client, "format_medchem_memory", None)
        if not callable(formatter):
            return ""
        try:
            return str(formatter() or "").strip()
        except Exception:
            return ""

    def _stream_chunk_text(self, chunk: Any) -> str:
        content = getattr(chunk, "content", None)
        if content is None and isinstance(chunk, dict):
            content = chunk.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") in {None, "text"}:
                        parts.append(str(item.get("text") or item.get("content") or ""))
            return "".join(parts)
        return ""

    async def _build_main_agent(self) -> Any:
        """Build the tool-calling main agent."""
        if self._main_agent is None:
            tools = await self._get_main_agent_tools()
            backend = FilesystemBackend(
                root_dir=self.config.deepagent.backend_root_dir,
                virtual_mode=True,
            )
            self._main_agent = create_deep_agent(
                model=create_base_model(self.config.llm),
                system_prompt=(
                    self._uploaded_documents_full_text_prompt()
                    + self._uploaded_structure_context_prompt()
                    + self._uploaded_table_context_prompt()
                    + MAIN_AGENT_SYSTEM_PROMPT
                    + self._additional_context_prompt()
                    + self._document_session_prompt()
                ),
                tools=tools,
                skills=None,
                backend=backend,
                middleware=[
                    *get_middleware(self.config.llm),
                    MainAgentToolExclusionMiddleware(MAIN_AGENT_EXCLUDED_TOOL_NAMES),
                ],
            )
        return self._main_agent

    def _additional_context_prompt(self) -> str:
        """Expose startup context to the main controller as read-only context."""
        context = self.main_additional_context
        if (
            context is None
            and not self.table_context
            and not self.structure_context
            and not self.uploaded_document_context
        ):
            context = self.additional_context
        if not context:
            return ""
        return (
            "\n[SESSION ADDITIONAL CONTEXT]\n"
            "The following startup context has been injected into the specialist "
            "agents as task-specific reference material. Use it when relevant, "
            "and do not invent unsupported values or conclusions.\n"
            f"{context}\n"
        )

    def _uploaded_structure_context_prompt(self) -> str:
        """Place uploaded SDF/PDB conversions directly in the main prompt."""
        if not self.structure_context:
            return ""
        return (
            "[UPLOADED STRUCTURE FILE CONTEXT]\n"
            "The following user-provided SDF/PDB files have been processed into "
            "machine-ready ligand SMILES and/or protein FASTA context. Use these "
            "values directly when they are relevant, while keeping file boundaries clear.\n\n"
            f"{self.structure_context}\n"
            "[END UPLOADED STRUCTURE FILE CONTEXT]\n\n"
        )

    def _uploaded_table_context_prompt(self) -> str:
        """Place uploaded table context directly in the main controller prompt."""
        if not self.table_context:
            return ""
        return (
            "[UPLOADED TABLE FULL CONTEXT]\n"
            "The following user-provided table data has been loaded in full. "
            "Use these table values directly when they are relevant, while keeping "
            "source and sheet boundaries intact.\n\n"
            f"{self.table_context}\n"
            "[END UPLOADED TABLE FULL CONTEXT]\n\n"
        )

    def _document_session_prompt(self) -> str:
        """Build dynamic system prompt context for uploaded documents."""
        if not self.document_kb_id:
            return ""
        info = self.knowledge_context.get_uploaded_file_info()
        documents = info.get("documents", [])
        if not documents:
            return ""
        names = [item.get("source_name") or item.get("markdown_name") for item in documents]
        names_text = "\n".join(f"- {name}" for name in names if name)
        return (
            "\n[UPLOADED DOCUMENT SESSION STATE]\n"
            f"This chat session has already preprocessed {len(documents)} uploaded document(s) "
            f"into converted Markdown context set {self.document_kb_id}.\n"
            f"Total converted Markdown characters: {info.get('total_chars', 0)}. "
            f"Documents:\n{names_text}\n"
            "The converted full text is included at the top of this system prompt. "
            "Use that Markdown block directly for document questions.\n"
        )

    def _uploaded_documents_full_text_prompt(self) -> str:
        """Place converted uploaded-document Markdown before all other instructions."""
        if self.uploaded_document_context:
            return f"{self.uploaded_document_context}\n\n"
        if not self.document_kb_id:
            return ""
        info = self.knowledge_context.get_uploaded_file_info()
        documents = info.get("documents", [])
        if not documents:
            return ""

        sections: List[str] = []
        loaded_names: List[str] = []
        for item in documents:
            markdown_file = item.get("markdown_file")
            if not markdown_file:
                continue
            path = Path(markdown_file)
            if not path.exists() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                continue
            source_name = item.get("source_name") or item.get("markdown_name") or path.name
            loaded_names.append(str(source_name))
            sections.append(
                "\n".join(
                    [
                        f"## Document: {source_name}",
                        f"Converted Markdown file: {path.name}",
                        "",
                        text,
                    ]
                )
            )

        if not sections:
            return ""

        names_text = ", ".join(loaded_names)
        return (
            "[UPLOADED DOCUMENT FULL TEXT - CONVERTED MARKDOWN]\n"
            "The following uploaded documents were converted to Markdown before this "
            "agent session started. Treat this block as the authoritative uploaded-document "
            "context for the current session. Use it directly; vector RAG and chunk retrieval "
            "are disabled for uploaded documents in this mode.\n"
            f"Covered documents: {names_text}\n\n"
            + "\n\n---\n\n".join(sections)
            + "\n[END UPLOADED DOCUMENT FULL TEXT]\n\n"
        )

    async def _get_main_agent_tools(self) -> List[Any]:
        """Return MCP tools plus local orchestration tools for the main agent."""
        if self._main_agent_tools is not None:
            return self._main_agent_tools

        excluded_mcp_tools = {"smiles_to_fragments_str"}
        if not self.config.enable_molecular_auxiliary_context:
            excluded_mcp_tools.add("generate_iupac_name")
        mcp_tools = self.mcp_client.get_tools_by_names(TOOL_NAMES_MCP - excluded_mcp_tools)

        @tool("smiles_to_fragments_str")
        async def smiles_to_fragments_str_tool(smiles_list: List[str]) -> Dict[str, str]:
            """Fragment molecule SMILES strings into functional-group names via MCP, with local fallback."""
            try:
                return await self.mcp_client.invoke_tool(
                    tool_name="smiles_to_fragments_str",
                    params={"smiles_list": smiles_list},
                )
            except AttributeError:
                from m3os.agents_v5.tools.fragments import smiles2fragments_str

                return smiles2fragments_str(smiles_list)

        @tool("get_uploaded_file_info")
        async def get_uploaded_file_info_tool() -> str:
            """Inspect uploaded documents available in the current chat session."""
            return json.dumps(
                self.knowledge_context.get_uploaded_file_info(),
                ensure_ascii=False,
                indent=2,
            )

        @tool("ask_medchem_knowledge")
        async def ask_medchem_knowledge_tool(
            question: str,
            decision_context: str = "",
            answer_focus: str = "",
        ) -> str:
            """Ask the dedicated medicinal-chemistry retrieval agent for concise guidance."""
            if self.medchem_retrieval_agent is None:
                return json.dumps(
                    {
                        "answer": "Dedicated medicinal-chemistry retrieval agent is not configured.",
                        "key_points": [],
                        "sources": [],
                        "limitations": ["ask_medchem_knowledge is unavailable in this session."],
                        "memory_hit": False,
                    },
                    ensure_ascii=False,
                )
            return await self.medchem_retrieval_agent.answer(
                question=question,
                decision_context=decision_context,
                answer_focus=answer_focus,
                requester_role="main",
                requester_context={},
            )

        @tool("run_mcgs_search")
        async def run_mcgs_search_tool(
            project_manager_brief: str,
            user_request: Optional[str] = None,
            max_expansion_rounds: Optional[int] = None,
            generator_mode: Optional[str] = None,
            target_candidate_count: Optional[int] = None,
            base_smiles: Optional[str] = None,
            refresh_task: bool = False,
        ) -> str:
            """Run MCGS optimization only for candidate generation/search.

            Call this directly for optimization/search requests. This tool runs
            the selected-candidate analysis needed by the MCGS workflow.

            Args:
                project_manager_brief: Required project-manager instruction for
                    the MCGS optimization team. Include the molecule,
                    explicitly requested goal, explicit constraints, requested
                    candidate count or rounds, explicitly requested ADMET
                    priorities, and relevant uploaded/document context cues.
                    Do not add inferred secondary property objectives such as
                    logP, solubility, TPSA, hERG, clearance, permeability, or
                    synthetic feasibility unless the user explicitly requested
                    them or they are hard constraints in uploaded/table context.
                user_request: Latest user request or refinement for this search.
                max_expansion_rounds: Set this ONLY when the user explicitly asks
                    for a number of MCGS/search/optimization rounds or iterations.
                    Do not infer this from the requested number of molecules.
                    When set, the search executes exactly this many expansion
                    rounds unless an expansion fails.
                generator_mode: Optional explicit generator switch. Use
                    "rational" to run only Rational Designer, "creative" to run
                    only Creative Explorer, or "both" for the default.
                target_candidate_count: Set this when the user requests a
                    concrete number of candidate molecules. In multi-turn
                    searches this requests that many new valid non-root
                    candidates for this tool call.
                base_smiles: Set this when the user explicitly chooses a
                    molecule/SMILES/candidate as the base for the next
                    optimization. If the molecule is not already in the
                    current graph, M3OS attaches it below the most similar
                    existing graph node and expands from the inserted molecule.
                refresh_task: Set true when the latest user request changes the
                    optimization goal, hard constraints, ADMET priorities, or
                    initial molecule. This rebuilds the project-manager brief
                    and frozen ADMET policy. If base_smiles is provided, the
                    current search graph is preserved and expanded from that
                    molecule, inserting it under the most similar graph node
                    when needed; otherwise a refreshed task creates a new graph.
            """
            return await self._run_mcgs_search_from_tool(
                project_manager_brief=project_manager_brief,
                user_request=user_request,
                max_expansion_rounds=max_expansion_rounds,
                generator_mode=generator_mode,
                target_candidate_count=target_candidate_count,
                base_smiles=base_smiles,
                refresh_task=refresh_task,
            )

        @tool("generate_optimization_report")
        async def generate_optimization_report_tool(
            output_dir: str = "tmp/reports",
            filename: Optional[str] = None,
            top_n: int = 8,
            synthesis_shortlist_n: int = 5,
            coverage_mode: str = "all",
            max_detailed_cards: Optional[int] = None,
            style_reference: str = "m3os_evolution_atlas",
            report_language: Optional[str] = None,
            report_request_text: str = "",
        ) -> str:
            """Generate a new self-contained HTML report only when the user explicitly asks for a report file."""
            if self.state is None:
                return (
                    "No prepared MCGS optimization state is available. "
                    "Please run candidate generation/search before generating the report."
                )
            report_generator = getattr(self, "report_generator", None)
            if report_generator is None:
                report_generator = KimiCodeReportGenerator(config=self.config)
                self.report_generator = report_generator
            result = await report_generator.generate(
                graph_snapshot=self.get_graph_snapshot(),
                state=self.state,
                output_dir=output_dir,
                filename=filename,
                top_n=top_n,
                synthesis_shortlist_n=synthesis_shortlist_n,
                coverage_mode=coverage_mode,
                max_detailed_cards=max_detailed_cards,
                style_reference=style_reference,
                report_language=report_language,
                report_request_text=report_request_text,
            )
            self.tool_cache["last_report_path"] = result.output_path
            self.tool_cache["last_report_html"] = result.html
            self.state["optimization_report_path"] = result.output_path
            self.state["optimization_report_html"] = result.html
            return (
                "Kimi Code HTML optimization report generated.\n"
                f"Path: {result.output_path}\n"
                f"Report workspace: {result.workspace_path}\n"
                f"Coverage mode: {coverage_mode}\n"
                f"Top molecule focus target: {top_n}\n"
                f"Synthesis shortlist size: {synthesis_shortlist_n}"
            )

        raw_tools = [*mcp_tools]
        if self.config.enable_molecular_auxiliary_context:
            raw_tools.append(smiles_to_fragments_str_tool)
        raw_tools.extend(
            [
                get_uploaded_file_info_tool,
                run_mcgs_search_tool,
                generate_optimization_report_tool,
            ]
        )
        if self.medchem_retrieval_agent is not None:
            raw_tools.insert(-2, ask_medchem_knowledge_tool)
        self._main_agent_tools = [self._wrap_tool_with_trace(tool_item) for tool_item in raw_tools]
        return self._main_agent_tools

    def _wrap_tool_with_trace(self, tool_item: Any) -> Any:
        """Print CLI traces and emit web events whenever the main agent invokes a tool."""
        tool_name = getattr(tool_item, "name", str(tool_item))
        description = getattr(tool_item, "description", "") or ""
        args_schema = getattr(tool_item, "args_schema", None)
        return_direct = getattr(tool_item, "return_direct", False)

        async def traced_tool(**kwargs: Any) -> Any:
            return await self._execute_traced_tool(tool_item, tool_name, kwargs)

        return StructuredTool.from_function(
            coroutine=traced_tool,
            name=tool_name,
            description=description,
            args_schema=args_schema,
            return_direct=return_direct,
        )

    async def _execute_traced_tool(
        self,
        tool_item: Any,
        tool_name: str,
        kwargs: Dict[str, Any],
    ) -> Any:
        self._raise_if_cancelled()
        started_at = time.perf_counter()
        await self._emit_event(
            "tool_call",
            "M3OS",
            self._format_tool_call_event(tool_name),
        )
        print(
            f"\n[M3OS tool call] {tool_name} args={self._summarize_tool_payload(kwargs)}",
            flush=True,
        )
        try:
            self._raise_if_cancelled()
            result = await self._invoke_wrapped_tool(tool_item, tool_name, kwargs)
            self._raise_if_cancelled()
        except asyncio.CancelledError:
            elapsed = time.perf_counter() - started_at
            await self._emit_event(
                "m3os_tool_summary",
                "M3OS",
                self._format_tool_result_event(
                    tool_name,
                    None,
                    elapsed=elapsed,
                    error=self._cancel_reason or "M3OS run stopped by user.",
                ),
            )
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            await self._emit_event(
                "m3os_tool_summary",
                "M3OS",
                self._format_tool_result_event(
                    tool_name,
                    None,
                    elapsed=elapsed,
                    error=f"{type(exc).__name__}: {exc}",
                ),
            )
            print(
                f"[M3OS tool error] {tool_name} failed after {elapsed:.2f}s: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if self.mcp_client.is_connection_error(exc):
                return self._format_tool_failure_result(tool_name, exc)
            raise

        elapsed = time.perf_counter() - started_at
        await self._emit_event(
            "m3os_tool_summary",
            "M3OS",
            self._format_tool_result_event(tool_name, result, elapsed=elapsed),
        )
        print(
            f"[M3OS tool result] {tool_name} completed in {elapsed:.2f}s; "
            f"result={self._summarize_tool_payload(result)}",
            flush=True,
        )
        return result

    async def _invoke_wrapped_tool(
        self,
        tool_item: Any,
        tool_name: str,
        kwargs: Dict[str, Any],
    ) -> Any:
        if self.mcp_client.has_tool(tool_name):
            return await self.mcp_client.invoke_tool(tool_name, kwargs)
        return await tool_item.ainvoke(kwargs)

    def _format_tool_failure_result(self, tool_name: str, exc: BaseException) -> Dict[str, Any]:
        return {
            "status": "failed",
            "tool": tool_name,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "retryable": True,
        }

    def _format_tool_call_event(self, tool_name: str) -> Dict[str, Any]:
        metadata = {
            "run_mcgs_search": (
                "MCGS search",
                "search",
                "Running multi-agent molecular graph search.",
            ),
            "generate_optimization_report": (
                "Report generation",
                "report",
                "Preparing an optimization report from current search results.",
            ),
            "get_uploaded_file_info": (
                "Uploaded files",
                "knowledge",
                "Checking uploaded files available to this thread.",
            ),
            "get_protein_uniprot_ids": (
                "Protein ID lookup",
                "knowledge",
                "Resolving target protein names to UniProt accessions, optionally by species taxonomy ID.",
            ),
            "get_protein_sequence": (
                "Protein sequence lookup",
                "knowledge",
                "Fetching protein FASTA sequence from UniProt.",
            ),
            "get_molecule_smiles": (
                "Molecule SMILES lookup",
                "knowledge",
                "Resolving molecule or drug names to SMILES.",
            ),
            "smiles_to_fragments_str": (
                "Fragment analysis",
                "analysis",
                "Identifying functional fragments from SMILES.",
            ),
        }
        label, stage, message = metadata.get(
            tool_name,
            (tool_name.replace("_", " ").title(), "tool", f"Running {tool_name}."),
        )
        return {
            "tool": tool_name,
            "label": label,
            "stage": stage,
            "message": message,
        }

    def _format_tool_result_event(
        self,
        tool_name: str,
        result: Any,
        *,
        elapsed: float,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "tool": tool_name,
            "label": self._format_tool_call_event(tool_name)["label"],
            "stage": self._format_tool_call_event(tool_name)["stage"],
            "status": "failed" if error else "completed",
            "elapsed_seconds": round(elapsed, 2),
            "summary": "",
            "display": {},
            "molecules": [],
            "metrics": {},
        }
        if error:
            payload["summary"] = error
            payload["display"] = {"error": error}
            return payload

        result_text = str(result or "")
        if tool_name == "run_mcgs_search":
            snapshot = self.get_graph_snapshot()
            summary = snapshot.get("summary") or {}
            payload["summary"] = self._compact_text(result_text, 360)
            payload["molecules"] = (
                self._critic_auditor_molecules_for_return()
                or self._top_snapshot_molecules(snapshot, limit=1000)
            )
            payload["display"] = {
                "search_summary": self._compact_text(result_text, 700),
                "best_smiles": snapshot.get("best_smiles"),
                "current_smiles": snapshot.get("current_smiles"),
            }
            payload["metrics"] = {
                "node_count": summary.get("node_count"),
                "edge_count": summary.get("edge_count"),
                "finished": summary.get("finished"),
            }
            return payload

        if tool_name == "generate_optimization_report":
            payload["summary"] = self._compact_text(result_text, 360)
            payload["display"] = {"report": self._compact_text(result_text, 900)}
            if self.tool_cache.get("last_report_path"):
                payload["display"]["path"] = self.tool_cache.get("last_report_path")
            return payload

        payload["summary"] = self._compact_text(result_text, 360)
        payload["display"] = {"preview": self._compact_text(result_text, 900)}
        return payload

    async def _emit_event(self, event_type: str, source: str, content: Any) -> None:
        event_sink = getattr(self, "event_sink", None)
        if event_sink is None:
            return
        payload = {
            "type": event_type,
            "source": source,
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        result = event_sink(payload)
        if hasattr(result, "__await__"):
            await result

    async def _emit_agent_message(
        self,
        *,
        message: str,
        agent_type: str,
        agent_name: str,
        stage: str = "reasoning",
        round_number: Optional[int] = None,
        max_rounds: Optional[int] = None,
    ) -> None:
        text = self._compact_text(message, 1800)
        if not text:
            return
        await self._emit_event(
            "m3os_agent_message",
            agent_name,
            {
                "agent_type": agent_type,
                "agent_name": agent_name,
                "stage": stage,
                "message": text,
                "round": round_number,
                "display_round": round_number,
                "max_rounds": max_rounds,
            },
        )

    async def _emit_workflow_step(
        self,
        node: str,
        message: str,
        *,
        round_number: Optional[int] = None,
        max_rounds: Optional[int] = None,
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._emit_event(
            "m3os_workflow_step",
            "M3OS",
            {
                "node": node,
                "stage": stage or node,
                "message": message,
                "round": round_number,
                "display_round": round_number,
                "max_rounds": max_rounds,
                "details": self._json_safe(details or {}),
            },
        )

    async def _emit_candidate_selected(
        self,
        current_info: Dict[str, Any],
        *,
        round_number: Optional[int] = None,
        max_rounds: Optional[int] = None,
        shared_analysis: Optional[Dict[str, Any]] = None,
    ) -> None:
        smiles = str(current_info.get("current_smiles") or "")
        if not smiles:
            return
        select_reason = current_info.get("current_select_reason")
        is_initial_selection = (
            round_number == 1
            and "initial molecule" in str(select_reason or "").lower()
        ) or "no selection needed" in str(select_reason or "").lower()
        analysis_payload = self._shared_analysis_event_payload(shared_analysis)
        await self._emit_event(
            "m3os_candidate_selected",
            "M3OS",
            {
                "agent_type": "main_agent",
                "round": round_number,
                "display_round": round_number,
                "max_rounds": max_rounds,
                "smiles": smiles,
                "iupac": current_info.get("current_iupac"),
                "fragments": current_info.get("current_fragments"),
                "reason": select_reason,
                "selection_context": current_info.get("current_selection_context"),
                "analysis": analysis_payload,
                "shared_analysis": analysis_payload,
                "analysis_summary": analysis_payload.get("shared_analysis_summary", ""),
                "is_initial_selection": is_initial_selection,
                "message": f"Selected {smiles} for the next MCGS expansion.",
            },
        )

    def _shared_analysis_event_payload(self, shared_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(shared_analysis, dict):
            return {}
        fields = {
            "shared_analysis_summary": shared_analysis.get("shared_analysis_summary", ""),
            "shared_keep_fragments": shared_analysis.get("shared_keep_fragments", []),
            "shared_modifiable_fragments": shared_analysis.get("shared_modifiable_fragments", []),
            "shared_risk_alerts": shared_analysis.get("shared_risk_alerts", []),
            "shared_priority_directions": shared_analysis.get("shared_priority_directions", []),
        }
        return {key: self._compact_analysis_value(value) for key, value in fields.items()}

    def _compact_analysis_value(self, value: Any, *, depth: int = 0) -> Any:
        if isinstance(value, str):
            return self._truncate_text(value, 3500 if depth == 0 else 1200)
        if isinstance(value, dict):
            if depth >= 3:
                return self._truncate_text(json.dumps(value, ensure_ascii=False, default=str), 1200)
            items = list(value.items())[:12]
            return {
                str(key): self._compact_analysis_value(item, depth=depth + 1)
                for key, item in items
            }
        if isinstance(value, (list, tuple, set)):
            return [
                self._compact_analysis_value(item, depth=depth + 1)
                for item in list(value)[:12]
            ]
        return self._json_safe(value)

    @staticmethod
    def _truncate_text(value: Any, max_chars: int) -> str:
        text = str(value or "").replace("\r", "\n").strip()
        if len(text) <= max_chars:
            return text
        omitted = len(text) - max_chars
        return f"{text[:max_chars].rstrip()}\n\n[truncated {omitted} chars]"

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return self._json_safe(model_dump(mode="json"))
            except TypeError:
                return self._json_safe(model_dump())
        return str(value)

    @staticmethod
    def _summarize_tool_payload(payload: Any, max_chars: int = 500) -> str:
        """Render tool arguments/results compactly so CLI traces stay readable."""
        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            text = str(payload)
        text = " ".join(text.split())
        if len(text) > max_chars:
            return f"{text[:max_chars]}...<truncated {len(text) - max_chars} chars>"
        return text

    @staticmethod
    def _compact_text(value: Any, max_chars: int = 500) -> str:
        return str(value or "").replace("\r", "\n").strip()

    async def get_main_agent_tool_names(self) -> List[str]:
        """Expose main-agent tool names for smoke tests and debugging."""
        tools = await self._get_main_agent_tools()
        return [getattr(tool, "name", str(tool)) for tool in tools]

    async def _run_mcgs_search_from_tool(
        self,
        *,
        project_manager_brief: str,
        user_request: Optional[str] = None,
        max_expansion_rounds: Optional[int] = None,
        generator_mode: Optional[str] = None,
        target_candidate_count: Optional[int] = None,
        base_smiles: Optional[str] = None,
        refresh_task: bool = False,
    ) -> str:
        self._raise_if_cancelled()
        if generator_mode:
            enabled = self.set_generator_mode(generator_mode)
            await self._emit_event(
                "assistant_update",
                "M3OS",
                {
                    "message": f"Generator mode set to: {', '.join(enabled)}.",
                    "enabled_generators": enabled,
                },
            )

        try:
            graph_base_smiles = self._resolve_existing_graph_base_smiles(base_smiles)
        except ValueError as exc:
            return str(exc)

        if self.state is None:
            prepared = await self.prepare_task(
                user_request=user_request,
                project_manager_brief=project_manager_brief,
                run_initial_analysis=False,
            )
            self._raise_if_cancelled()
            try:
                search_summary = await self.run_search_until_target(
                    max_expansion_rounds=max_expansion_rounds,
                    target_candidate_count=target_candidate_count,
                    base_smiles=None,
                )
            except ValueError as exc:
                return f"{prepared}\n\n{exc}"
            return f"{prepared}\n\n{search_summary}"

        if refresh_task and graph_base_smiles:
            prepared = await self.refresh_existing_graph_task_context(
                user_request=user_request,
                project_manager_brief=project_manager_brief,
                base_smiles=graph_base_smiles,
            )
            self._raise_if_cancelled()
            try:
                search_summary = await self.run_search_until_target(
                    max_expansion_rounds=max_expansion_rounds,
                    target_candidate_count=target_candidate_count,
                    base_smiles=graph_base_smiles,
                )
            except ValueError as exc:
                return f"{prepared}\n\n{exc}"
            return f"{prepared}\n\n{search_summary}"

        if refresh_task:
            prepared = await self.prepare_task(
                user_request=user_request,
                project_manager_brief=project_manager_brief,
                run_initial_analysis=False,
            )
            self._raise_if_cancelled()
            try:
                search_summary = await self.run_search_until_target(
                    max_expansion_rounds=max_expansion_rounds,
                    target_candidate_count=target_candidate_count,
                    base_smiles=None,
                )
            except ValueError as exc:
                return f"{prepared}\n\n{exc}"
            return f"{prepared}\n\n{search_summary}"

        self._update_project_manager_brief_for_run(project_manager_brief or user_request)
        try:
            return await self.run_search_until_target(
                max_expansion_rounds=max_expansion_rounds,
                target_candidate_count=target_candidate_count,
                base_smiles=graph_base_smiles or base_smiles,
            )
        except ValueError as exc:
            return str(exc)

    def _task_preparation_prompt(
        self,
        *,
        user_request: Optional[str] = None,
        project_manager_brief: Optional[str] = None,
        continuation_base_smiles: Optional[str] = None,
    ) -> str:
        prompt = self._conversation_text()
        if project_manager_brief:
            prompt = f"{prompt}\nproject_manager_brief: {project_manager_brief}"
        if user_request:
            prompt = f"{prompt}\nuser_request_for_search: {user_request}"
        if continuation_base_smiles:
            prompt = (
                f"{prompt}\ncontinuation_base_smiles: {continuation_base_smiles}"
                "\nExisting graph continuation: preserve the current Molecular Search Graph "
                "and refresh only task-level context for this base molecule."
            )
        return prompt

    async def prepare_task(
        self,
        user_request: Optional[str] = None,
        project_manager_brief: Optional[str] = None,
        *,
        run_initial_analysis: bool = True,
    ) -> str:
        """Prepare or refresh optimization state from the full conversation."""
        self._raise_if_cancelled()
        prompt = self._task_preparation_prompt(
            user_request=user_request,
            project_manager_brief=project_manager_brief,
        )
        updates = await self.task_preparation.prepare_initial_state(
            user_prompt=prompt,
            project_manager_brief=project_manager_brief,
            additional_context=self.additional_context,
            run_initial_analysis=run_initial_analysis,
        )
        state = MCGSState(
            user_prompt=prompt,
            additional_context=self.additional_context,
            current_shared_analysis={},
            shared_analysis_cache={},
            molecule_facts_cache={},
            agent_memory={"rational": [], "creative": [], "critic": []},
            screening_context_history=[],
            creative_agent_traces=[],
            rational_agent_traces=[],
            subagent_trace_rounds=[],
            critic_auditor_evaluations=[],
            runtime_metrics=[],
        )
        state.update(updates)
        state["protein_fasta_path"] = self.protein_fasta_path
        state["leadopt_task_contract"] = dict(self.leadopt_task_contract)
        self.state = state
        self._remember_prepared_state()
        self._raise_if_cancelled()
        await self._emit_event(
            "assistant_update",
            "M3OS",
            {
                "message": "Prepared molecular optimization task state.",
                "initial_smiles": self.state.get("initial_smiles"),
                "optimization_goal": self.state.get("optimization_goal"),
                "project_manager_brief": self.state.get("project_manager_brief"),
                "target_candidates": self.state.get("node_num_needed"),
                "target_iterations": self.state.get("iteration_num_needed"),
            },
        )
        await self._emit_graph_snapshot("Task prepared")
        return "MCGS Round Finish."

    async def refresh_existing_graph_task_context(
        self,
        *,
        user_request: Optional[str] = None,
        project_manager_brief: Optional[str] = None,
        base_smiles: str,
    ) -> str:
        """Refresh task-level context while preserving the current MCGS graph."""
        self._raise_if_cancelled()
        if self.state is None:
            return await self.prepare_task(
                user_request=user_request,
                project_manager_brief=project_manager_brief,
                run_initial_analysis=False,
            )
        graph_before = self.state.get("mcgs_graph")
        root_smiles = str(self.state.get("initial_smiles") or "")
        prompt = self._task_preparation_prompt(
            user_request=user_request,
            project_manager_brief=project_manager_brief,
            continuation_base_smiles=base_smiles,
        )
        updates = await self.task_preparation.prepare_existing_graph_context(
            user_prompt=prompt,
            base_smiles=base_smiles,
            project_manager_brief=project_manager_brief,
            additional_context=self.additional_context,
        )
        facts_cache = dict(self.state.get("molecule_facts_cache") or {})
        new_facts = updates.pop("molecule_facts_cache", {})
        if isinstance(new_facts, dict):
            facts_cache.update(new_facts)
            facts_by_smiles = self._cache_bucket("molecule_facts_by_smiles")
            for key, facts in new_facts.items():
                if isinstance(facts, dict):
                    facts_by_smiles[str(key)] = facts
                    smiles = str(facts.get("smiles") or "")
                    if smiles:
                        facts_by_smiles[smiles] = facts

        for key, value in updates.items():
            self.state[key] = value
        self.state["molecule_facts_cache"] = facts_cache
        self.state["mcgs_graph"] = graph_before
        if root_smiles:
            self.state["initial_smiles"] = root_smiles
        self._raise_if_cancelled()
        await self._emit_event(
            "assistant_update",
            "M3OS",
            {
                "message": "Refreshed molecular optimization task context while preserving the search graph.",
                "initial_smiles": self.state.get("initial_smiles"),
                "base_smiles": base_smiles,
                "optimization_goal": self.state.get("optimization_goal"),
                "project_manager_brief": self.state.get("project_manager_brief"),
                "target_candidates": self.state.get("node_num_needed"),
                "target_iterations": self.state.get("iteration_num_needed"),
            },
        )
        await self._emit_graph_snapshot("Task context refreshed")
        return (
            "Task context refreshed without resetting the Molecular Search Graph. "
            f"Root SMILES: {self.state.get('initial_smiles', 'N/A')}; "
            f"base SMILES: {base_smiles}; "
            f"goal: {self.state.get('optimization_goal', 'N/A')}; "
            f"target candidate molecules: {self.state.get('node_num_needed', 'N/A')}; "
            f"target MCGS iterations: {self.state.get('iteration_num_needed') or 'auto'}."
        )

    async def run_search_until_target(
        self,
        max_expansion_rounds: Optional[int] = None,
        target_candidate_count: Optional[int] = None,
        base_smiles: Optional[str] = None,
    ) -> str:
        """Run main-agent-controlled MCGS expansions until the target is met."""
        self._raise_if_cancelled()
        if self.state is None:
            await self.prepare_task(run_initial_analysis=False)
        assert self.state is not None

        if not self.state.get("mcgs_graph"):
            self._raise_if_cancelled()
            await self._emit_workflow_step(
                "init_graph",
                "Initializing the molecular search graph.",
                stage="graph_initialization",
            )
            updates = await self.init_graph_node(self.state)
            self._merge_updates(updates)
            await self._emit_graph_snapshot("Initialized molecular search graph")

        rounds_run = 0
        explicit_round_limit = self._resolve_requested_rounds(max_expansion_rounds)
        if explicit_round_limit is None:
            state_round_limit = self._state_iteration_round_limit()
            if not (
                target_candidate_count is not None
                and self.state.get("iteration_num_defaulted")
            ):
                explicit_round_limit = state_round_limit
        stop_on_candidate_target = explicit_round_limit is None
        fixed_base_smiles = self._resolve_fixed_base_smiles(
            base_smiles,
        )
        run_target = self._resolve_run_target_candidate_count(target_candidate_count)
        baseline_count = self._candidate_node_count()
        self.state["run_candidate_baseline"] = baseline_count
        self.state["run_target_candidate_count"] = run_target
        self.state["run_fixed_base_smiles"] = fixed_base_smiles
        if target_candidate_count is not None:
            self.state["node_num_needed"] = run_target
        max_rounds = explicit_round_limit or self._resolve_max_expansion_rounds(
            target_candidate_count=run_target,
            baseline_count=baseline_count,
        )
        max_no_progress_rounds = self._max_no_progress_rounds()
        no_progress_rounds = 0

        while rounds_run < max_rounds:
            self._raise_if_cancelled()
            if stop_on_candidate_target and self._run_candidate_target_reached(baseline_count, run_target):
                break
            before_round_candidate_count = self._candidate_node_count()
            trace_cursor = self._trace_cursor()
            await self._emit_event(
                "mcgs_round_started",
                "M3OS",
                {
                    "round": rounds_run + 1,
                    "max_rounds": max_rounds,
                    "candidate_count": self._candidate_node_count(),
                    "target_candidates": run_target,
                    "candidate_baseline": baseline_count,
                    "new_candidate_count": self._new_candidate_count(baseline_count),
                    "fixed_base_smiles": fixed_base_smiles,
                    "round_limit_source": "user" if explicit_round_limit is not None else "auto",
                    "message": f"Starting MCGS expansion round {rounds_run + 1}.",
                },
            )
            self._raise_if_cancelled()
            use_fixed_base = bool(fixed_base_smiles and rounds_run == 0)
            if use_fixed_base:
                current_info = await self._build_fixed_base_current_info(fixed_base_smiles)
                self._merge_updates({"current_info_list": [current_info]})
                await self._emit_workflow_step(
                    "select_candidate",
                    "Using the user-specified molecule as the base for this expansion.",
                    round_number=rounds_run + 1,
                    max_rounds=max_rounds,
                    stage="selection",
                    details={"fixed_base_smiles": fixed_base_smiles},
                )
            elif self._graph_has_only_root():
                current_info = (self.state.get("current_info_list") or [{}])[-1]
                await self._emit_workflow_step(
                    "select_candidate",
                    "Using the prepared initial molecule as the first expansion base.",
                    round_number=rounds_run + 1,
                    max_rounds=max_rounds,
                    stage="selection",
                )
            else:
                await self._emit_workflow_step(
                    "select_candidate",
                    "Selecting the next molecule to expand from the MCGS graph.",
                    round_number=rounds_run + 1,
                    max_rounds=max_rounds,
                    stage="selection",
                )
                select_updates = await self.select_candidate_node(self.state)
                self._merge_updates(select_updates)
            current_info = (self.state.get("current_info_list") or [{}])[-1]
            await self._emit_graph_snapshot("Selected next candidate node")

            self._raise_if_cancelled()
            await self._emit_workflow_step(
                "analyze_current",
                "Analyzing the selected molecule before generator agents propose analogs.",
                round_number=rounds_run + 1,
                max_rounds=max_rounds,
                stage="analysis",
            )
            analysis_updates = await self.analyze_current_node(self.state)
            self._merge_updates(analysis_updates)
            shared = self.state.get("current_shared_analysis") or {}
            await self._emit_event(
                "assistant_update",
                "M3OS",
                {
                    "message": "Analyzed current molecular candidate.",
                    "current_smiles": self._current_smiles(),
                },
            )
            await self._emit_candidate_selected(
                current_info,
                round_number=rounds_run + 1,
                max_rounds=max_rounds,
                shared_analysis=shared,
            )

            await self._emit_workflow_step(
                "expansion_workflow",
                "Running Rational Designer, Creative Explorer, and Critic for this expansion.",
                round_number=rounds_run + 1,
                max_rounds=max_rounds,
                stage="multi_agent_expansion",
            )
            self.state["current_mcgs_round"] = rounds_run + 1
            await self._run_expansion_round(
                round_number=rounds_run + 1,
                max_rounds=max_rounds,
                trace_cursor=trace_cursor,
            )
            self._raise_if_cancelled()
            rounds_run += 1
            await self._emit_workflow_step(
                "graph_update",
                "Updating the molecular search graph with evaluated candidates.",
                round_number=rounds_run,
                max_rounds=max_rounds,
                stage="graph_update",
            )
            await self._emit_graph_snapshot("Expanded molecular search graph")

            after_round_candidate_count = self._candidate_node_count()
            if after_round_candidate_count <= before_round_candidate_count:
                no_progress_rounds += 1
            else:
                no_progress_rounds = 0

            if stop_on_candidate_target and self._run_candidate_target_reached(baseline_count, run_target):
                break
            if (
                explicit_round_limit is None
                and max_no_progress_rounds > 0
                and no_progress_rounds >= max_no_progress_rounds
            ):
                await self._emit_event(
                    "assistant_update",
                    "M3OS",
                    {
                        "message": (
                            "Stopping automatic MCGS expansion because recent rounds "
                            "did not add new valid candidates."
                        ),
                        "new_candidate_count": self._new_candidate_count(baseline_count),
                        "target_candidates": run_target,
                        "no_progress_rounds": no_progress_rounds,
                    },
                )
                break

        reached_run_target = self._run_candidate_target_reached(baseline_count, run_target)
        if (stop_on_candidate_target and reached_run_target) or (
            explicit_round_limit is not None and rounds_run >= max_rounds
        ):
            self.state["is_finished"] = True
        else:
            self.state["is_finished"] = False

        self.last_search_result = self.state
        await self._emit_workflow_step(
            "search_complete",
            "MCGS search complete.",
            round_number=rounds_run,
            max_rounds=max_rounds,
            stage="complete",
            details={
                "finished": self.state.get("is_finished", False),
                "new_candidate_count": self._new_candidate_count(baseline_count),
                "target_candidates": run_target,
                "candidate_baseline": baseline_count,
            },
        )
        await self._emit_graph_snapshot("MCGS search complete")
        return self._search_summary(rounds_run)

    def _conversation_text(self) -> str:
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in self.messages
        )

    def _new_tool_cache(self) -> Dict[str, Any]:
        return {
            "molecule_facts_by_smiles": {},
        }

    def _cache_bucket(self, name: str) -> Dict[str, Any]:
        bucket = self.tool_cache.setdefault(name, {})
        if not isinstance(bucket, dict):
            bucket = {}
            self.tool_cache[name] = bucket
        return bucket

    async def _get_molecule_facts(
        self,
        smiles: str,
        iupac: Optional[Any] = None,
        fragments: Optional[Any] = None,
        need_iupac: bool = False,
        need_fragments: bool = False,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        facts_by_smiles = self._cache_bucket("molecule_facts_by_smiles")
        key = self._canonical_smiles_key(smiles)
        facts = dict(facts_by_smiles.get(key, facts_by_smiles.get(smiles, {})))

        if not self.config.enable_molecular_auxiliary_context:
            facts = {
                "smiles": smiles,
                "canonical_smiles": key,
                "iupac": "",
                "fragments": "",
            }
            facts_by_smiles[key] = facts
            if self.state is not None:
                state_cache = dict(self.state.get("molecule_facts_cache") or {})
                state_cache[key] = facts
                self.state["molecule_facts_cache"] = state_cache
            return facts

        if iupac is not None:
            facts["iupac"] = iupac
        if fragments is not None:
            facts["fragments"] = fragments

        if need_iupac and (force_refresh or not facts.get("iupac")):
            facts["iupac"] = await self.task_preparation.get_iupac(smiles)
        if need_fragments and (force_refresh or not facts.get("fragments")):
            facts["fragments"] = await self.task_preparation.get_fragments(smiles)

        facts["smiles"] = smiles
        facts["canonical_smiles"] = key
        facts_by_smiles[key] = facts
        if self.state is not None:
            state_cache = dict(self.state.get("molecule_facts_cache") or {})
            state_cache[key] = facts
            self.state["molecule_facts_cache"] = state_cache
        return facts

    async def _build_fixed_base_current_info(self, smiles: str) -> Dict[str, Any]:
        facts = await self._get_molecule_facts(
            smiles,
            need_iupac=True,
            need_fragments=True,
        )
        attachment = {}
        if self.state and isinstance(self.state.get("last_external_base_attachment"), dict):
            latest_attachment = self.state.get("last_external_base_attachment") or {}
            if latest_attachment.get("base_smiles") == smiles:
                attachment = latest_attachment
        selection_context = f"Continue optimization from the user-specified base molecule: {smiles}"
        select_reason = "User specified this molecule as the base for the next optimization."
        if attachment:
            selection_context += (
                "\nThis valid user-specified molecule was not already in the graph, so M3OS "
                "attached it below the most similar existing graph node "
                f"{attachment.get('parent_smiles')} "
                f"(Tanimoto similarity {attachment.get('parent_similarity')})."
            )
            select_reason = (
                "User specified this external molecule as the base; M3OS inserted it "
                "under the most similar existing graph node before expansion."
            )
        return {
            "current_smiles": smiles,
            "current_iupac": facts.get("iupac") or (
                "None" if self.config.enable_molecular_auxiliary_context else ""
            ),
            "current_fragments": facts.get("fragments") or (
                "None" if self.config.enable_molecular_auxiliary_context else ""
            ),
            "current_selection_context": selection_context,
            "current_select_reason": select_reason,
        }

    def _resolve_existing_graph_base_smiles(self, base_smiles: Optional[str]) -> str:
        if not self.state or not self.state.get("mcgs_graph"):
            return ""
        return self._resolve_fixed_base_smiles(base_smiles)

    def _resolve_fixed_base_smiles(
        self,
        base_smiles: Optional[str],
    ) -> str:
        text = str(base_smiles or "").strip()
        if not text:
            return ""
        ok, canonical, error = validate_smiles(text)
        if not ok:
            raise ValueError(
                "Invalid base SMILES. Please provide a valid molecular SMILES string."
                f" RDKit validation failed: {error}"
        )
        graph_smiles = self._find_graph_smiles(text, canonical)
        if not graph_smiles:
            graph_smiles = self._attach_external_base_smiles(
                canonical or text,
                raw_smiles=text,
            )
        return graph_smiles

    def _find_graph_smiles(self, raw_smiles: str, canonical_smiles: str = "") -> str:
        if not self.state or not self.state.get("mcgs_graph"):
            return ""
        graph = self.state["mcgs_graph"]
        for candidate in (raw_smiles, canonical_smiles):
            if candidate and hasattr(graph, "get_node_by_smiles"):
                try:
                    node = graph.get_node_by_smiles(candidate)
                except Exception:
                    node = None
                if node is not None:
                    return str(getattr(node, "smiles", "") or candidate)
        target_key = canonical_smiles or self._canonical_smiles_key(raw_smiles)
        try:
            nodes = list(graph.get_all_nodes()) if hasattr(graph, "get_all_nodes") else []
        except Exception:
            nodes = []
        for node in nodes:
            smiles = str(getattr(node, "smiles", "") or "")
            if smiles and self._canonical_smiles_key(smiles) == target_key:
                return smiles
        return ""

    def _attach_external_base_smiles(self, canonical_smiles: str, *, raw_smiles: str = "") -> str:
        """Attach a valid user-specified external base below its closest graph node."""
        if not self.state or not self.state.get("mcgs_graph"):
            return ""
        graph = self.state["mcgs_graph"]
        if not hasattr(graph, "get_all_nodes") or not hasattr(graph, "add_node") or not hasattr(graph, "add_edge"):
            raise ValueError("The current M3OS search graph does not support inserting an external base molecule.")

        smiles = str(canonical_smiles or raw_smiles or "").strip()
        if not smiles:
            return ""

        try:
            nodes = [
                node for node in list(graph.get_all_nodes())
                if str(getattr(node, "smiles", "") or "").strip()
            ]
        except Exception:
            nodes = []
        if not nodes:
            raise ValueError("The current M3OS search graph is empty; an external base molecule cannot be attached.")

        candidate_smiles = [str(getattr(node, "smiles", "") or "") for node in nodes]
        try:
            from m3os.agents_v5.tools.molecular_similarity import batch_tanimoto_similarity_search

            hits = batch_tanimoto_similarity_search(smiles, candidate_smiles, max_workers=1)
        except Exception as exc:
            raise ValueError(f"Failed to find similar graph nodes for the external base molecule: {exc}") from exc
        if not hits:
            raise ValueError("The current M3OS search graph has no valid molecule nodes for similarity matching.")

        parent_smiles = hits[0].smiles
        parent_node = graph.get_node_by_smiles(parent_smiles) if hasattr(graph, "get_node_by_smiles") else None
        if parent_node is None:
            parent_node = next(
                (node for node in nodes if str(getattr(node, "smiles", "") or "") == parent_smiles),
                None,
            )
        if parent_node is None:
            raise ValueError("Could not locate the most similar parent node in the current M3OS search graph.")

        properties: Dict[str, Any] = {
            "inserted_by_user": True,
            "external_base_smiles": smiles,
            "raw_user_smiles": str(raw_smiles or smiles),
            "nearest_graph_parent_smiles": parent_smiles,
            "nearest_graph_parent_similarity": round(float(hits[0].similarity), 4),
        }
        root_similarity = tanimoto_similarity_to_root(smiles, self.state.get("initial_smiles"))
        if root_similarity is not None:
            properties["root_similarity"] = root_similarity

        parent_score = self._number_or_none(getattr(parent_node, "intrinsic_score", None)) or 0.0
        iteration = int(getattr(graph, "iteration_count", 0) or 0)
        node = MoleculeNode(
            smiles=smiles,
            intrinsic_score=parent_score,
            properties=properties,
            iteration=iteration,
            agent_type="human_insert",
        )
        action = "User-selected external base molecule attached to nearest graph neighbor."
        node.parents.append(parent_node)
        node.actions_from_parents[parent_node.smiles] = action
        node.rationale_from_parents_generator[parent_node.smiles] = (
            "Inserted because the user chose this valid SMILES as the next optimization base; "
            f"nearest existing graph node by Morgan Tanimoto similarity was {parent_smiles} "
            f"(similarity={properties['nearest_graph_parent_similarity']})."
        )
        node.rationale_from_parents_critic[parent_node.smiles] = ""
        node.confidence_score_from_parents_generator[parent_node.smiles] = 1.0
        graph.add_node(node)
        graph.add_edge(MoleculeEdge(parent_node, node, action))

        attachment = {
            "base_smiles": smiles,
            "raw_user_smiles": str(raw_smiles or smiles),
            "parent_smiles": parent_smiles,
            "parent_similarity": properties["nearest_graph_parent_similarity"],
            "root_similarity": root_similarity,
        }
        self.state["last_external_base_attachment"] = attachment
        self.state["external_base_insertions"] = list(self.state.get("external_base_insertions", [])) + [attachment]
        return smiles

    def _graph_has_only_root(self) -> bool:
        if not self.state or not self.state.get("mcgs_graph"):
            return False
        graph = self.state["mcgs_graph"]
        try:
            nodes = list(graph.get_all_nodes()) if hasattr(graph, "get_all_nodes") else []
        except Exception:
            return False
        return len(nodes) <= 1

    @staticmethod
    def _canonical_smiles_key(smiles: Any) -> str:
        ok, canonical, _error = validate_smiles(smiles)
        return canonical if ok and canonical else str(smiles or "").strip()

    def _remember_prepared_state(self) -> None:
        if not self.state:
            return
        smiles = self.state.get("initial_smiles")
        if smiles:
            key = self._canonical_smiles_key(smiles)
            facts_by_smiles = self._cache_bucket("molecule_facts_by_smiles")
            facts = {
                "smiles": smiles,
                "canonical_smiles": key,
                "iupac": self.state.get("initial_iupac"),
                "fragments": self.state.get("initial_fragments"),
            }
            facts_by_smiles[key] = facts
            state_cache = dict(self.state.get("molecule_facts_cache") or {})
            state_cache[key] = facts
            self.state["molecule_facts_cache"] = state_cache
            self.tool_cache["last_smiles"] = smiles
            self.tool_cache["last_iupac"] = self.state.get("initial_iupac")
            self.tool_cache["last_fragments"] = self.state.get("initial_fragments")

    def _update_project_manager_brief_for_run(self, user_request: Optional[str]) -> None:
        """Carry the latest main-agent instruction into an existing MCGS search."""
        if not self.state:
            return
        latest = str(user_request or "").strip()
        if not latest:
            return
        current = str(self.state.get("project_manager_brief") or "").strip()
        if latest in current:
            return
        if current:
            updated = f"{current}\n\n[Latest Project Manager Instruction]\n{latest}"
        else:
            updated = latest
        self.state["project_manager_brief"] = self._truncate_text(updated, 12000)

    def _search_summary(self, rounds_run: int) -> str:
        graph_nodes = self._graph_node_count()
        current_info = (self.state.get("current_info_list") or [{}])[-1] if self.state else {}
        baseline = int(self.state.get("run_candidate_baseline") or 0) if self.state else 0
        run_target = int(self.state.get("run_target_candidate_count") or self._target_candidate_count()) if self.state else 0
        new_candidates = self._new_candidate_count(baseline)
        return (
            "MCGS search complete.\n"
            f"Expansion rounds run: {rounds_run}\n"
            f"New candidate molecules this run: {new_candidates}/{run_target}\n"
            f"Candidate molecules in graph: {self._candidate_node_count()}\n"
            f"Graph nodes: {graph_nodes}\n"
            f"Current molecule: {current_info.get('current_smiles', 'N/A')}\n"
            f"Finished: {self.state.get('is_finished', False) if self.state else False}"
            f"{self._critic_auditor_summary_block()}"
        )

    def _critic_auditor_summary_block(self) -> str:
        molecules = self._critic_auditor_molecules_for_return()
        if not molecules:
            return ""
        lines = ["", "", "Auditor structured critic evaluations:"]
        for index, molecule in enumerate(molecules, start=1):
            properties = molecule.get("properties") if isinstance(molecule.get("properties"), dict) else {}
            properties_text = json.dumps(properties, ensure_ascii=False, sort_keys=True) if properties else "{}"
            lines.extend(
                [
                    f"{index}. Round: {molecule.get('round', 'N/A')}",
                    f"   SMILES: {molecule.get('smiles', '')}",
                    f"   Action: {molecule.get('action', '')}",
                    f"   Score: {molecule.get('score', 0.0)}",
                    f"   Rationale: {molecule.get('rationale', '')}",
                    f"   Properties: {properties_text}",
                    f"   Agent type: {molecule.get('agent_type', 'critic')}",
                ]
            )
        return "\n".join(lines)

    def _critic_auditor_molecules_for_return(self) -> List[Dict[str, Any]]:
        if not self.state:
            return []
        traces = self.state.get("critic_auditor_evaluations") or []
        result: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            round_label = trace.get("display_round") or trace.get("round_index") or ""
            structured = trace.get("structured_response")
            raw_items = trace.get("optimizations")
            items = raw_items if isinstance(raw_items, list) else structured_optimizations(structured)
            for item in items:
                smiles = str(structured_field(item, "smiles", "") or "").strip()
                if not smiles:
                    continue
                ok, canonical, _error = validate_smiles(smiles)
                if not ok:
                    continue
                dedupe_key = canonical or smiles
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                raw_properties = structured_field(item, "properties", {})
                properties = raw_properties if isinstance(raw_properties, dict) else {}
                result.append(
                    {
                        "round": round_label,
                        "smiles": smiles,
                        "action": str(structured_field(item, "action", "") or ""),
                        "rationale": str(structured_field(item, "rationale", "") or ""),
                        "score": self._number_or_none(structured_field(item, "score", 0.0)) or 0.0,
                        "properties": self._json_safe(properties),
                        "agent_type": str(structured_field(item, "agent_type", "critic") or "critic"),
                    }
                )
        return result

    def _resolve_run_target_candidate_count(self, target_candidate_count: Optional[int]) -> int:
        if target_candidate_count is not None:
            try:
                return max(1, int(target_candidate_count))
            except (TypeError, ValueError):
                return self._target_candidate_count()
        return self._target_candidate_count()

    def _new_candidate_count(self, baseline_count: int) -> int:
        return max(0, self._candidate_node_count() - int(baseline_count or 0))

    def _run_candidate_target_reached(self, baseline_count: int, target_candidate_count: int) -> bool:
        return self._new_candidate_count(baseline_count) >= max(1, int(target_candidate_count or 1))

    def _max_auto_expansion_rounds(self) -> int:
        try:
            value = int(os.environ.get("M3OS_MAX_AUTO_EXPANSION_ROUNDS", "25") or 25)
        except (TypeError, ValueError):
            value = 25
        return max(1, value)

    def _max_no_progress_rounds(self) -> int:
        try:
            value = int(os.environ.get("M3OS_MAX_NO_PROGRESS_ROUNDS", "2") or 2)
        except (TypeError, ValueError):
            value = 2
        return max(0, value)

    async def _run_expansion_round(
        self,
        *,
        round_number: int,
        max_rounds: int,
        trace_cursor: Dict[str, Any],
    ) -> None:
        if self.state is None:
            return
        self._raise_if_cancelled()
        stage_state = {"stage": "starting", "generated": 0, "evaluated": 0}
        if "creative" in self.enabled_generators:
            await self._emit_subagent_started(
                "creative_molecule_explorer",
                "Creative Explorer is generating analogs.",
                round_number,
                max_rounds,
            )
        if "rational" in self.enabled_generators:
            await self._emit_subagent_started(
                "rational_medicinal_designer",
                "Rational Designer is integrating medicinal chemistry knowledge and cases.",
                round_number,
                max_rounds,
            )
        self.state["optimized_molecules_creative"] = []
        self.state["optimized_molecules_rational"] = []
        heartbeat_task = asyncio.create_task(
            self._round_heartbeat_loop(round_number, max_rounds, stage_state)
        )
        used_stream = False
        try:
            astream = getattr(self.expansion_workflow, "astream", None)
            if not callable(astream):
                raise RuntimeError("Compiled MCGS workflow does not expose astream.")
            async for update in astream(
                self.state,
                config=self._langgraph_config(),
                stream_mode="updates",
            ):
                self._raise_if_cancelled()
                used_stream = True
                await self._handle_expansion_stream_update(
                    update,
                    round_number=round_number,
                    max_rounds=max_rounds,
                    stage_state=stage_state,
                )
        except Exception as exc:
            if used_stream:
                await self._emit_event(
                    "assistant_update",
                    "M3OS",
                    {
                        "message": f"MCGS streaming interrupted: {type(exc).__name__}: {exc}",
                        "round": round_number,
                    },
                )
                raise
            await self._emit_event(
                "assistant_update",
                "M3OS",
                {
                    "message": "MCGS streaming is unavailable in this runtime; continuing with compatible execution.",
                    "round": round_number,
                },
            )
            self._raise_if_cancelled()
            self.state = await self.expansion_workflow.ainvoke(
                self.state,
                config=self._langgraph_config(),
            )
            self._raise_if_cancelled()
            await self._emit_new_subagent_events(trace_cursor)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _handle_expansion_stream_update(
        self,
        update: Any,
        *,
        round_number: int,
        max_rounds: int,
        stage_state: Dict[str, Any],
    ) -> None:
        self._raise_if_cancelled()
        if not isinstance(update, dict):
            return
        for node_name, node_update in update.items():
            self._raise_if_cancelled()
            if not isinstance(node_update, dict):
                continue
            if node_name == "creative_explorer":
                stage_state["stage"] = "creative_explorer"
                self._merge_updates(node_update)
                traces = node_update.get("creative_agent_traces") or []
                stage_state["generated"] = len(node_update.get("optimized_molecules_creative") or [])
                await self._emit_workflow_step(
                    "creative_explorer",
                    "Creative Explorer generated broad analog candidates.",
                    round_number=round_number,
                    max_rounds=max_rounds,
                    stage="generation",
                    details={"generated_count": stage_state["generated"]},
                )
                for trace in traces:
                    payload = self._format_generation_trace(trace, "creative_molecule_explorer")
                    await self._emit_agent_message(
                        message=payload.get("summary") or payload.get("title") or "",
                        agent_type="creative_molecule_explorer",
                        agent_name="Creative Explorer",
                        stage="generation",
                        round_number=round_number,
                        max_rounds=max_rounds,
                    )
                    await self._emit_event("m3os_subagent_result", "Creative Explorer", payload)
                continue
            if node_name == "rational_designer":
                stage_state["stage"] = "rational_designer"
                self._merge_updates(node_update)
                traces = node_update.get("rational_agent_traces") or []
                generated = len(node_update.get("optimized_molecules_rational") or [])
                stage_state["generated"] = int(stage_state.get("generated") or 0) + generated
                await self._emit_workflow_step(
                    "rational_designer",
                    "Rational Designer generated medicinal-chemistry-guided candidates.",
                    round_number=round_number,
                    max_rounds=max_rounds,
                    stage="generation",
                    details={"generated_count": generated},
                )
                for trace in traces:
                    payload = self._format_generation_trace(trace, "rational_medicinal_designer")
                    await self._emit_agent_message(
                        message=payload.get("summary") or payload.get("title") or "",
                        agent_type="rational_medicinal_designer",
                        agent_name="Rational Designer",
                        stage="generation",
                        round_number=round_number,
                        max_rounds=max_rounds,
                    )
                    await self._emit_event("m3os_subagent_result", "Rational Designer", payload)
                continue
            if node_name == "molecules_aggregator":
                stage_state["stage"] = "molecules_aggregator"
                self._merge_updates(node_update)
                molecules = node_update.get("optimized_molecules") or self.state.get("optimized_molecules") or []
                stage_state["generated"] = len(molecules)
                await self._emit_workflow_step(
                    "molecules_aggregator",
                    f"Assembled {len(molecules)} candidate molecule{'s' if len(molecules) != 1 else ''} for critic evaluation.",
                    round_number=round_number,
                    max_rounds=max_rounds,
                    stage="aggregation",
                    details={"candidate_count": len(molecules)},
                )
                await self._emit_event(
                    "m3os_candidate_batch",
                    "M3OS",
                    self._format_candidate_batch(molecules, round_number, node_update),
                )
                await self._emit_workflow_step(
                    "critic",
                    "Critic is evaluating candidate molecules.",
                    round_number=round_number,
                    max_rounds=max_rounds,
                    stage="evaluation",
                    details={"candidate_count": len(molecules)},
                )
                await self._emit_event(
                    "m3os_critic_started",
                    "Critic",
                    {
                        "agent_type": "critic",
                        "round": round_number,
                        "display_round": round_number,
                        "max_rounds": max_rounds,
                        "candidate_count": len(molecules),
                        "message": "Critic is evaluating candidates.",
                    },
                )
                continue
            if node_name == "evaluate_and_update":
                stage_state["stage"] = "evaluate_and_update"
                self._merge_updates(node_update)
                critic = node_update.get("critic_evaluation") or self.state.get("critic_evaluation") or []
                stage_state["evaluated"] = len(critic)
                molecules = [self._format_critic_molecule(item) for item in critic if isinstance(item, dict)]
                await self._emit_workflow_step(
                    "evaluate_and_update",
                    f"Critic evaluated {len(molecules)} candidate molecule{'s' if len(molecules) != 1 else ''} and updated scores.",
                    round_number=round_number,
                    max_rounds=max_rounds,
                    stage="evaluation",
                    details={"evaluated_count": len(molecules)},
                )
                await self._emit_event(
                    "m3os_critic_evaluation",
                    "Critic",
                    {
                        "agent_type": "critic",
                        "round": round_number,
                        "display_round": round_number,
                        "title": f"Critic evaluated {len(molecules)} candidate molecule{'s' if len(molecules) != 1 else ''}",
                        "molecules": molecules,
                        "invalid_molecules_filtered": node_update.get("invalid_molecules_filtered", 0),
                        "invalid_molecules": node_update.get("invalid_molecules", []),
                        "failed": bool(node_update.get("expansion_error")),
                        "error": str(node_update.get("expansion_error") or ""),
                    },
                )
                await self._emit_graph_snapshot("Critic updated molecular search graph")
                continue
            self._merge_updates(node_update)

    async def _emit_subagent_started(
        self,
        agent_type: str,
        message: str,
        round_number: int,
        max_rounds: int,
    ) -> None:
        await self._emit_event(
            "m3os_subagent_started",
            "M3OS",
            {
                "agent_type": agent_type,
                "round": round_number,
                "display_round": round_number,
                "max_rounds": max_rounds,
                "message": message,
            },
        )

    async def _round_heartbeat_loop(
        self,
        round_number: int,
        max_rounds: int,
        stage_state: Dict[str, Any],
    ) -> None:
        started_at = time.perf_counter()
        while True:
            await asyncio.sleep(6)
            counts = self._graph_counts()
            await self._emit_event(
                "m3os_round_heartbeat",
                "M3OS",
                {
                    "round": round_number,
                    "display_round": round_number,
                    "max_rounds": max_rounds,
                    "stage": stage_state.get("stage") or "running",
                    "elapsed_seconds": round(time.perf_counter() - started_at, 1),
                    "generated_count": stage_state.get("generated") or 0,
                    "evaluated_count": stage_state.get("evaluated") or 0,
                    "node_count": counts.get("nodes"),
                    "edge_count": counts.get("edges"),
                },
            )

    def _graph_counts(self) -> Dict[str, Any]:
        if not self.state or not self.state.get("mcgs_graph"):
            return {"nodes": 0, "edges": 0}
        graph = self.state["mcgs_graph"]
        try:
            nodes = list(graph.get_all_nodes()) if hasattr(graph, "get_all_nodes") else []
        except Exception:
            nodes = []
        edge_count = 0
        if hasattr(graph, "get_parents"):
            for node in nodes:
                try:
                    edge_count += len(list(graph.get_parents(node)))
                except Exception:
                    pass
        return {"nodes": len(nodes), "edges": edge_count}

    def _format_candidate_batch(
        self,
        molecules: Any,
        round_number: int,
        invalid_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        safe_molecules = []
        for item in molecules or []:
            if not isinstance(item, dict):
                continue
            safe_molecules.append(
                self._format_generated_molecule(
                    {
                        "smiles": item.get("smiles") or item.get("SMILES"),
                        "modification_type": item.get("modification_type") or item.get("action"),
                        "action": item.get("action") or item.get("modification_type"),
                        "rationale": item.get("rationale"),
                        "confidence_score": item.get("confidence_score"),
                        "properties": item.get("properties") or {},
                    },
                    str(item.get("agent_type") or "candidate"),
                )
            )
        return {
            "agent_type": "candidate_batch",
            "round": round_number,
            "display_round": round_number,
            "title": f"Candidate batch assembled with {len(safe_molecules)} molecule{'s' if len(safe_molecules) != 1 else ''}",
            "molecules": safe_molecules,
            "invalid_molecules_filtered": (invalid_context or {}).get("invalid_molecules_filtered", 0),
            "invalid_molecules": (invalid_context or {}).get("invalid_molecules", []),
            "failed": False,
        }

    def _trace_cursor(self) -> Dict[str, Any]:
        if not self.state:
            return {}
        critic = self.state.get("critic_evaluation") or []
        return {
            "creative": len(self.state.get("creative_agent_traces") or []),
            "rational": len(self.state.get("rational_agent_traces") or []),
            "subagent_rounds": len(self.state.get("subagent_trace_rounds") or []),
            "critic_signature": json.dumps(self._json_safe(critic), ensure_ascii=False, sort_keys=True, default=str),
        }

    async def _emit_new_subagent_events(self, cursor: Dict[str, Any]) -> None:
        if not self.state:
            return
        creative_traces = self.state.get("creative_agent_traces") or []
        rational_traces = self.state.get("rational_agent_traces") or []
        for trace in creative_traces[int(cursor.get("creative", 0)):]:
            await self._emit_event(
                "m3os_subagent_result",
                "Creative Explorer",
                self._format_generation_trace(trace, "creative_molecule_explorer"),
            )
        for trace in rational_traces[int(cursor.get("rational", 0)):]:
            await self._emit_event(
                "m3os_subagent_result",
                "Rational Designer",
                self._format_generation_trace(trace, "rational_medicinal_designer"),
            )

        critic = self.state.get("critic_evaluation") or []
        critic_signature = json.dumps(self._json_safe(critic), ensure_ascii=False, sort_keys=True, default=str)
        if critic and critic_signature != cursor.get("critic_signature"):
            molecules = [self._format_critic_molecule(item) for item in critic if isinstance(item, dict)]
            await self._emit_event(
                "m3os_critic_evaluation",
                "Critic",
                {
                    "agent_type": "critic",
                    "title": f"Critic evaluated {len(molecules)} candidate molecule{'s' if len(molecules) != 1 else ''}",
                    "molecules": molecules,
                    "invalid_molecules_filtered": self.state.get("invalid_molecules_filtered", 0),
                    "invalid_molecules": self.state.get("invalid_molecules", []),
                    "failed": False,
                },
            )

    def _format_generation_trace(self, trace: Any, fallback_agent_type: str) -> Dict[str, Any]:
        data = trace if isinstance(trace, dict) else {}
        agent_type = str(data.get("agent_type") or fallback_agent_type)
        raw_molecules = [
            item for item in data.get("processed_molecules") or []
            if isinstance(item, dict)
        ]
        valid_molecules, invalid_molecules = split_valid_molecules(raw_molecules)
        molecules = [
            self._format_generated_molecule(item, agent_type)
            for item in valid_molecules
        ]
        agent_label = (
            "Creative Explorer"
            if "creative" in agent_type
            else "Rational Designer"
        )
        display_round = (
            data.get("display_mcgs_round")
            or data.get("mcgs_round")
            or self._display_round(data.get("round_index"))
        )
        summary_parts = [
            f"{agent_label} proposed {len(molecules)} analog{'s' if len(molecules) != 1 else ''}."
        ]
        rationales = [
            item.get("rationale")
            for item in molecules[:2]
            if isinstance(item, dict) and item.get("rationale")
        ]
        if rationales:
            summary_parts.append("Key rationale: " + " ".join(str(item) for item in rationales))
        return {
            "agent_type": agent_type,
            "round_index": data.get("round_index"),
            "mcgs_round": data.get("mcgs_round") or display_round,
            "display_round": display_round,
            "display_mcgs_round": data.get("display_mcgs_round") or display_round,
            "title": f"{agent_label} proposed {len(molecules)} analog{'s' if len(molecules) != 1 else ''}",
            "summary": self._compact_text(" ".join(summary_parts)),
            "molecules": molecules,
            "invalid_molecules_filtered": len(invalid_molecules),
            "invalid_molecules": compact_invalid_molecules(invalid_molecules),
            "raw_agent_response": self._compact_text(data.get("raw_agent_response"), 4000),
            "audit_warnings": self._json_safe(data.get("audit_warnings") or []),
            "auditor_trace": self._json_safe(data.get("auditor_trace") or {}),
            "failed": bool(data.get("failed")),
            "error": str(data.get("error") or ""),
        }

    def _format_generated_molecule(self, item: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
        smiles = str(item.get("smiles") or item.get("SMILES") or "")
        return {
            "smiles": smiles,
            "label": self._compact_text(item.get("modification_type") or item.get("action") or "Generated analog", 80),
            "modification_type": str(item.get("modification_type") or ""),
            "action": str(item.get("action") or item.get("modification_type") or ""),
            "rationale": self._compact_text(item.get("rationale")),
            "generator_rationale": self._compact_text(item.get("rationale")),
            "confidence_score": self._number_or_none(item.get("confidence_score")),
            "properties": self._json_safe(item.get("properties") or {}),
            "agent_type": str(item.get("agent_type") or agent_type),
        }

    def _format_critic_molecule(self, item: Dict[str, Any]) -> Dict[str, Any]:
        rationale = (
            item.get("score_reason")
            or item.get("rationale")
            or item.get("critic_rationale")
            or ""
        )
        return {
            "smiles": str(item.get("smiles") or item.get("SMILES") or ""),
            "label": self._compact_text(item.get("action") or "Critic evaluation", 80),
            "action": str(item.get("action") or ""),
            "rationale": self._compact_text(item.get("rationale") or rationale),
            "critic_rationale": self._compact_text(rationale),
            "score_reason": self._compact_text(rationale),
            "score": self._number_or_none(item.get("score")),
            "properties": self._json_safe(item.get("properties") or {}),
            "agent_type": str(item.get("agent_type") or "critic"),
        }

    @staticmethod
    def _display_round(round_index: Any) -> int:
        try:
            return max(1, int(round_index or 1))
        except (TypeError, ValueError):
            return 1

    async def _emit_graph_snapshot(self, message: str) -> None:
        snapshot = self.get_graph_snapshot()
        if not snapshot.get("nodes"):
            return
        await self._emit_event(
            "graph_snapshot",
            "M3OS",
            {
                **snapshot,
                "message": message,
            },
        )

    def get_graph_snapshot(self) -> Dict[str, Any]:
        """Return a lightweight graph snapshot for web/event consumers.

        The FastAPI frontend enriches this with RDKit SVGs through its existing
        M3OS serializer. This method intentionally stays dependency-light so
        the agent can be used outside the web server too.
        """
        if not self.state or not self.state.get("mcgs_graph"):
            return {
                "updated_at": datetime.now(UTC).isoformat(),
                "root_smiles": self.state.get("initial_smiles", "") if self.state else "",
                "current_smiles": self._current_smiles(),
                "best_smiles": "",
                "nodes": [],
                "edges": [],
                "summary": self._snapshot_summary([], []),
            }

        graph = self.state["mcgs_graph"]
        root_node = getattr(graph, "root_node", None)
        root_id = str(getattr(root_node, "id", "") or "")
        root_smiles = str(getattr(root_node, "smiles", "") or self.state.get("initial_smiles", "") or "")
        current_smiles = self._current_smiles()
        raw_nodes = list(graph.get_all_nodes()) if hasattr(graph, "get_all_nodes") else []
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        for node in raw_nodes:
            node_id = str(getattr(node, "id", "") or "")
            smiles = str(getattr(node, "smiles", "") or "")
            parents = list(graph.get_parents(node)) if hasattr(graph, "get_parents") else []
            parent_ids = [str(getattr(parent, "id", "") or "") for parent in parents if getattr(parent, "id", None)]
            parent_smiles = [str(getattr(parent, "smiles", "") or "") for parent in parents if getattr(parent, "smiles", None)]
            preferred_keys = [item for pair in zip(parent_ids, parent_smiles) for item in pair if item]
            score = self._number_or_none(getattr(node, "intrinsic_score", None))
            properties = self._json_safe(getattr(node, "properties", {}) or {})
            properties = properties if isinstance(properties, dict) else {}
            node_payload = {
                "id": node_id,
                "smiles": smiles,
                "parent_ids": parent_ids,
                "parent_smiles": parent_smiles,
                "iteration": getattr(node, "iteration", 0) or 0,
                "agent_type": getattr(node, "agent_type", None) or "",
                "score": score,
                "critic_score": self._number_or_none(properties.get("critic_score")),
                "root_similarity": self._number_or_none(properties.get("root_similarity")),
                "mcgs_selection_score": self._number_or_none(properties.get("mcgs_selection_score")) or score,
                "total_reward": self._number_or_none(getattr(node, "total_reward", None)),
                "visit_count": int(getattr(node, "visit_count", 0) or 0),
                "uct_value": self._number_or_none(getattr(node, "uct_value", None)),
                "unreachable": bool(getattr(node, "unreachable", False)),
                "action": str(self._first_parent_value(getattr(node, "actions_from_parents", {}), preferred_keys) or ""),
                "generator_rationale": str(self._first_parent_value(getattr(node, "rationale_from_parents_generator", {}), preferred_keys) or ""),
                "critic_rationale": str(self._first_parent_value(getattr(node, "rationale_from_parents_critic", {}), preferred_keys) or ""),
                "confidence": self._number_or_none(self._first_parent_value(getattr(node, "confidence_score_from_parents_generator", {}), preferred_keys)),
                "properties": properties,
                "selected_reason": self._json_safe(getattr(node, "selected_reason", []) or []),
                "is_root": bool(node_id and node_id == root_id),
                "is_current": bool(smiles and current_smiles and smiles == current_smiles),
            }
            nodes.append(node_payload)
            for parent_id in parent_ids:
                edges.append(
                    {
                        "source": parent_id,
                        "target": node_id,
                        "action": node_payload["action"],
                        "score_delta": None,
                        "rationale_summary": self._compact_text(
                            node_payload["generator_rationale"] or node_payload["critic_rationale"],
                        ),
                    }
                )

        best_node = self._best_snapshot_node(nodes, root_id=root_id)
        best_id = best_node.get("id") or ""
        for node in nodes:
            node["is_best"] = bool(best_id and node.get("id") == best_id)

        return {
            "updated_at": datetime.now(UTC).isoformat(),
            "root_id": root_id,
            "root_smiles": root_smiles,
            "current_smiles": current_smiles,
            "best_smiles": best_node.get("smiles") or "",
            "best_node_id": best_id,
            "nodes": nodes,
            "edges": edges,
            "summary": self._snapshot_summary(nodes, edges),
        }

    def _current_smiles(self) -> str:
        if not self.state:
            return ""
        current_info = (self.state.get("current_info_list") or [{}])[-1]
        return str(current_info.get("current_smiles") or self.state.get("initial_smiles") or "")

    def _snapshot_summary(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = self._best_snapshot_node(nodes, root_id="")
        return {
            "root_smiles": self.state.get("initial_smiles", "") if self.state else "",
            "current_smiles": self._current_smiles(),
            "best_smiles": best.get("smiles") or "",
            "best_score": best.get("score"),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "finished": bool(self.state.get("is_finished")) if self.state else False,
            "optimization_goal": self.state.get("optimization_goal", "") if self.state else "",
        }

    @staticmethod
    def _number_or_none(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric != numeric:
            return None
        return numeric

    @staticmethod
    def _first_parent_value(mapping: Any, preferred_keys: List[str]) -> Any:
        if not isinstance(mapping, dict):
            return None
        for key in preferred_keys:
            if key in mapping:
                return mapping[key]
        for value in mapping.values():
            return value
        return None

    @staticmethod
    def _best_snapshot_node(nodes: List[Dict[str, Any]], root_id: str = "") -> Dict[str, Any]:
        candidates = [
            node for node in nodes
            if node.get("id") != root_id and not node.get("unreachable")
        ]
        if not candidates:
            candidates = [node for node in nodes if node.get("id") != root_id] or list(nodes)
        if not candidates:
            return {}
        return max(
            candidates,
            key=lambda node: (
                float(node.get("score") or 0.0),
                float(node.get("uct_value") or 0.0),
                int(node.get("visit_count") or 0),
            ),
        )

    def _top_snapshot_molecules(self, snapshot: Dict[str, Any], limit: int = 4) -> List[Dict[str, Any]]:
        nodes = [node for node in snapshot.get("nodes") or [] if isinstance(node, dict)]
        candidates = [node for node in nodes if not node.get("is_root")]
        candidates.sort(
            key=lambda node: (
                bool(node.get("unreachable")),
                -float(node.get("score") or 0.0),
                -float(node.get("uct_value") or 0.0),
            )
        )
        result = []
        for node in candidates[:limit]:
            result.append({
                "smiles": node.get("smiles") or "",
                "label": "Best candidate" if node.get("is_best") else self._compact_text(node.get("action") or node.get("agent_type") or "Candidate", 80),
                "score": node.get("score"),
                "critic_score": node.get("critic_score"),
                "root_similarity": node.get("root_similarity"),
                "mcgs_selection_score": node.get("mcgs_selection_score"),
                "action": node.get("action") or "",
                "rationale": self._compact_text(node.get("generator_rationale") or node.get("critic_rationale")),
                "properties": node.get("properties") or {},
                "agent_type": node.get("agent_type") or "",
            })
        return result

    def _graph_node_count(self) -> Any:
        if not self.state or not self.state.get("mcgs_graph"):
            return 0
        try:
            return len(self.state["mcgs_graph"].get_all_nodes())
        except Exception:
            return "N/A"

    def _candidate_node_count(self) -> int:
        if not self.state or not self.state.get("mcgs_graph"):
            return 0
        graph = self.state["mcgs_graph"]
        try:
            nodes = list(graph.get_all_nodes())
        except Exception:
            return 0
        root_node = getattr(graph, "root_node", None)
        root_id = str(getattr(root_node, "id", "") or "")
        root_smiles = str(getattr(root_node, "smiles", "") or self.state.get("initial_smiles") or "")
        count = 0
        for node in nodes:
            node_id = str(getattr(node, "id", "") or "")
            smiles = str(getattr(node, "smiles", "") or "")
            if root_id and node_id == root_id:
                continue
            if root_smiles and smiles == root_smiles:
                continue
            if bool(getattr(node, "unreachable", False)):
                continue
            count += 1
        return count

    def _target_candidate_count(self) -> int:
        if not self.state:
            return 0
        try:
            target = int(self.state.get("node_num_needed") or 0)
        except (TypeError, ValueError):
            target = 0
        return max(1, target)

    def _is_candidate_target_reached(self) -> bool:
        return self._candidate_node_count() >= self._target_candidate_count()

    def _resolve_requested_rounds(self, requested_rounds: Optional[int]) -> Optional[int]:
        if requested_rounds is not None:
            try:
                return max(1, int(requested_rounds))
            except (TypeError, ValueError):
                return None
        return None

    def _state_iteration_round_limit(self) -> Optional[int]:
        """Return the task-level requested MCGS iteration count, if any."""
        if not self.state:
            return None
        try:
            value = int(self.state.get("iteration_num_needed") or 0)
        except (TypeError, ValueError):
            return None
        return max(1, value) if value > 0 else None

    def _resolve_max_expansion_rounds(
        self,
        target_candidate_count: Optional[int] = None,
        baseline_count: Optional[int] = None,
    ) -> int:
        if target_candidate_count is None and baseline_count is None:
            target = self._target_candidate_count()
            current = self._candidate_node_count()
            remaining = max(0, target - current)
            if remaining <= 0:
                return 1
            expected_candidates_per_round = 5
            return max(1, min(target, (remaining + expected_candidates_per_round - 1) // expected_candidates_per_round + 1))
        return self._max_auto_expansion_rounds()

    @staticmethod
    def _normalize_recursion_limit(value: Any) -> int:
        try:
            parsed = int(value or MIN_LANGGRAPH_RECURSION_LIMIT)
        except (TypeError, ValueError):
            parsed = MIN_LANGGRAPH_RECURSION_LIMIT
        return max(MIN_LANGGRAPH_RECURSION_LIMIT, parsed)

    def _langgraph_config(self) -> Dict[str, int]:
        return {"recursion_limit": self._normalize_recursion_limit(self.recursion_limit)}

    def _merge_updates(self, updates: Dict[str, Any]) -> None:
        if self.state is None:
            self.state = MCGSState(**updates)
            return

        append_keys = {
            "current_info_list",
            "runtime_metrics",
            "creative_agent_traces",
            "rational_agent_traces",
            "subagent_trace_rounds",
            "critic_auditor_evaluations",
            "screening_context_history",
        }
        for key, value in updates.items():
            if key in append_keys:
                self.state[key] = list(self.state.get(key, [])) + list(value or [])
            else:
                self.state[key] = value

    def _extract_agent_response(self, result: Dict[str, Any]) -> str:
        messages = result.get("messages") if isinstance(result, dict) else None
        if not messages:
            return str(result)
        last = messages[-1]
        content = getattr(last, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "\n".join(part for part in text_parts if part).strip()
        return str(content or last)

    def _validate_main_agent_result(self, result: Any) -> bool:
        if not isinstance(result, dict):
            return bool(result)
        return bool(self._extract_agent_response(result).strip())


async def create_main_agent_session(
    env_file: Optional[str] = None,
    additional_context: Optional[str] = None,
    table_context_paths: Optional[List[str]] = None,
    structure_file_paths: Optional[List[str]] = None,
    protein_fasta_path: Optional[str] = None,
    leadopt_task_contract: Optional[Dict[str, Any]] = None,
    document_files: Optional[List[str]] = None,
    document_dir: Optional[str] = None,
    recursion_limit: int = 400,
    event_sink: Optional[AgentEventSink] = None,
    enabled_generators: Optional[List[str]] = None,
    enable_medchem_retrieval: bool = True,
    enable_molecular_auxiliary_context: bool = True,
    enable_optimization_case_retrieval: bool = True,
    kimi_reasoning_effort: Optional[str] = None,
) -> M3OSChatSession:
    """Factory function for the multi-turn main agent session."""
    return await M3OSChatSession.create(
        env_file=env_file,
        additional_context=additional_context,
        table_context_paths=table_context_paths,
        structure_file_paths=structure_file_paths,
        protein_fasta_path=protein_fasta_path,
        leadopt_task_contract=leadopt_task_contract,
        document_files=document_files,
        document_dir=document_dir,
        recursion_limit=recursion_limit,
        event_sink=event_sink,
        enabled_generators=enabled_generators,
        enable_medchem_retrieval=enable_medchem_retrieval,
        enable_molecular_auxiliary_context=enable_molecular_auxiliary_context,
        enable_optimization_case_retrieval=enable_optimization_case_retrieval,
        kimi_reasoning_effort=kimi_reasoning_effort,
    )
