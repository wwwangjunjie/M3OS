"""Knowledge context retrieval for agents_v5.

This service combines M3OS wiki MCP retrieval with an optional
preprocessed user-document knowledge base. It does not convert or index files;
that work must happen before chat or workflow execution starts.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.services.document_rag import DocumentRAGService
from m3os.agents_v5.services.mcp_client import WIKI_TOOL_NAMES
from m3os.agents_v5.services.m3os_wiki import (
    DEFAULT_WIKI_ARTICLE_MAX_CHARS,
    DEFAULT_WIKI_MAX_PAGES,
    M3OSWikiKnowledgeService,
)


class KnowledgeContextService:
    """Retrieve raw knowledge from built-in and preprocessed document stores."""

    def __init__(
        self,
        config: AgentConfig,
        document_kb_id: Optional[str] = None,
        document_kb_metadata: Optional[Dict[str, Any]] = None,
        mcp_client: Any = None,
    ):
        self.config = config
        self.document_kb_id = document_kb_id
        self.document_kb_metadata = document_kb_metadata or {}
        self.document_rag = DocumentRAGService(config.paths)
        self.mcp_client = mcp_client
        self.wiki_knowledge = self._build_wiki_service(mcp_client)

    def set_mcp_client(self, mcp_client: Any) -> None:
        """Attach or refresh the MCP client used for M3OS wiki retrieval."""
        self.mcp_client = mcp_client
        self.wiki_knowledge = self._build_wiki_service(mcp_client)

    async def retrieve_knowledge(
        self,
        query: str,
        task_context: Optional[str] = None,
        include_user_documents: bool = True,
        include_medchem: bool = True,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return raw knowledge and source metadata for one retrieval query."""
        medchem_context = ""
        sources: List[Dict[str, Any]] = []

        if include_medchem:
            if self.wiki_knowledge is None:
                medchem_context = (
                    "No M3OS wiki knowledge retrieved. "
                    "Reason: m3os_wiki_mcp_not_configured_or_unavailable"
                )
            else:
                wiki_payload = await self.wiki_knowledge.retrieve(
                    query=query,
                    task_context=task_context,
                )
                medchem_context = wiki_payload.get("content") or (
                    "No M3OS wiki knowledge retrieved."
                )
                sources.extend(wiki_payload.get("sources") or [])

        document_results: List[Dict[str, Any]] = []
        uploaded_document_rag_enabled = (
            self.document_kb_metadata.get("ingestion_mode") != "full_text_system_prompt"
            and self.document_kb_metadata.get("vector_store", True) is not False
        )
        if include_user_documents and self.document_kb_id and uploaded_document_rag_enabled:
            document_results = self.document_rag.retrieve_user_documents(
                query=query,
                kb_id=self.document_kb_id,
                top_k=top_k,
            )
            sources.extend(document_results)

        raw_knowledge = self.format_raw_knowledge(
            medchem_context=medchem_context,
            document_results=document_results,
            task_context=task_context,
        )
        return {
            "raw_knowledge": raw_knowledge,
            "filtered_knowledge": raw_knowledge,
            "sources": sources,
            "kb_id": self.document_kb_id,
            "topics": self.document_kb_metadata.get("topics", []),
        }

    def get_uploaded_file_info(self) -> Dict[str, Any]:
        """Return metadata for the current session's uploaded documents."""
        return self.document_rag.get_uploaded_file_info(self.document_kb_id)

    def retrieve_document_context(
        self,
        query: str,
        mode: str = "auto",
        document_names: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return user-document context using full text or vector retrieval."""
        info = self.get_uploaded_file_info()
        if not self.document_kb_id or not info.get("has_documents"):
            return {
                "mode": mode,
                "resolved_mode": "none",
                "intent": self.document_rag.classify_document_intent(query),
                "content": "No uploaded documents are available in this session.",
                "sources": [],
                "file_info": info,
                "reason": "no_uploaded_documents",
            }

        requested_mode = (mode or "auto").strip().lower()
        if requested_mode not in {"auto", "full", "rag"}:
            requested_mode = "auto"

        intent = self.document_rag.classify_document_intent(query)
        use_full = requested_mode == "full" or (
            requested_mode == "auto"
            and intent in {"global_summary", "comparative_summary", "mixed"}
            and bool(info.get("can_load_full_text"))
        )

        if use_full:
            payload = self.document_rag.load_uploaded_documents(
                kb_id=self.document_kb_id,
                document_names=document_names,
                max_chars=self.config.paths.rag_full_text_max_chars,
            )
            return {
                "mode": requested_mode,
                "resolved_mode": "full",
                "intent": intent,
                "content": payload.get("content", ""),
                "sources": payload.get("loaded_documents", []),
                "file_info": info,
                "truncated": payload.get("truncated", False),
                "reason": "summary_or_comparison_full_text",
            }

        reason = "explicit_rag"
        if requested_mode == "auto":
            reason = (
                "full_text_exceeds_limit"
                if intent in {"global_summary", "comparative_summary", "mixed"}
                else "fact_lookup_uses_rag"
            )

        document_results = self.document_rag.retrieve_user_documents(
            query=query,
            kb_id=self.document_kb_id,
            top_k=top_k,
        )
        content = self.format_document_results(document_results)
        if not content:
            content = "No matching uploaded-document chunks were retrieved."
        return {
            "mode": requested_mode,
            "resolved_mode": "rag",
            "intent": intent,
            "content": content,
            "sources": document_results,
            "file_info": info,
            "truncated": False,
            "reason": reason,
        }

    def format_raw_knowledge(
        self,
        medchem_context: str,
        document_results: Sequence[Dict[str, Any]],
        task_context: Optional[str],
    ) -> str:
        sections = []
        if task_context:
            sections.append(f"[TASK CONTEXT]\n{task_context}")
        if medchem_context:
            sections.append(f"[MEDICINAL CHEMISTRY KNOWLEDGE]\n{medchem_context}")
        if document_results:
            docs = []
            for index, item in enumerate(document_results, start=1):
                docs.append(
                    "Source {idx}: {metadata}\nScore: {score}\nContent: {content}".format(
                        idx=index,
                        metadata=item.get("metadata", {}),
                        score=item.get("score"),
                        content=item.get("content", ""),
                    )
                )
            sections.append("[USER DOCUMENT KNOWLEDGE]\n" + "\n\n".join(docs))
        if not sections:
            return "No specific knowledge retrieved."
        return "\n\n".join(sections)

    def format_document_results(self, document_results: Sequence[Dict[str, Any]]) -> str:
        """Format retrieved document chunks for LLM context."""
        docs = []
        for index, item in enumerate(document_results, start=1):
            docs.append(
                "Source {idx}: {metadata}\nScore: {score}\nContent: {content}".format(
                    idx=index,
                    metadata=item.get("metadata", {}),
                    score=item.get("score"),
                    content=item.get("content", ""),
                )
            )
        return "\n\n".join(docs)

    def _build_wiki_service(self, mcp_client: Any) -> Optional[M3OSWikiKnowledgeService]:
        if mcp_client is None:
            return None
        has_tool = getattr(mcp_client, "has_tool", None)
        if callable(has_tool) and not any(has_tool(tool_name) for tool_name in WIKI_TOOL_NAMES):
            return None
        return M3OSWikiKnowledgeService(
            mcp_client,
            max_pages=self._env_int("M3OS_WIKI_MAX_PAGES", DEFAULT_WIKI_MAX_PAGES),
            article_max_chars=self._env_int(
                "M3OS_WIKI_ARTICLE_MAX_CHARS",
                DEFAULT_WIKI_ARTICLE_MAX_CHARS,
            ),
        )

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)) or default)
        except (TypeError, ValueError):
            return default
