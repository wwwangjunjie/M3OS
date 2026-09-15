"""M3OS wiki MCP knowledge retrieval helpers."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional


DEFAULT_WIKI_MAX_PAGES = 5
DEFAULT_WIKI_ARTICLE_MAX_CHARS = 12000


class M3OSWikiKnowledgeService:
    """Retrieve medicinal chemistry knowledge from the M3OS synto MCP wiki."""

    def __init__(
        self,
        mcp_client: Any,
        *,
        max_pages: int = DEFAULT_WIKI_MAX_PAGES,
        article_max_chars: int = DEFAULT_WIKI_ARTICLE_MAX_CHARS,
    ) -> None:
        self.mcp_client = mcp_client
        self.max_pages = max(1, int(max_pages or DEFAULT_WIKI_MAX_PAGES))
        self.article_max_chars = max(1000, int(article_max_chars or DEFAULT_WIKI_ARTICLE_MAX_CHARS))

    async def retrieve(self, query: str, task_context: Optional[str] = None) -> Dict[str, Any]:
        """Return wiki knowledge payload for one medicinal chemistry query."""
        question = self._build_question(query=query, task_context=task_context)
        answer_payload = await self._answer_question(question)
        if answer_payload.get("content"):
            return answer_payload

        fallback_payload = await self._search_and_read(query)
        if fallback_payload.get("content"):
            return fallback_payload

        reason = answer_payload.get("reason") or fallback_payload.get("reason") or "no_wiki_result"
        return {
            "content": (
                "No M3OS wiki knowledge retrieved. "
                f"Reason: {reason}"
            ),
            "sources": [],
            "metadata": {"query": query, "reason": reason},
        }

    async def _answer_question(self, question: str) -> Dict[str, Any]:
        if not self._has_tool("answer_question"):
            return {"content": "", "sources": [], "reason": "answer_question_unavailable"}
        try:
            raw = await self.mcp_client.invoke_tool(
                "answer_question",
                {"question": question, "max_pages": self.max_pages},
            )
        except Exception as exc:
            return {
                "content": "",
                "sources": [],
                "reason": f"answer_question_failed: {type(exc).__name__}: {exc}",
            }

        data = self._coerce_structured(raw)
        if not isinstance(data, dict):
            text = self._stringify_result(raw).strip()
            if not text:
                return {"content": "", "sources": [], "reason": "answer_question_empty"}
            data = {"answer": text}

        answer = str(data.get("answer") or data.get("content") or "").strip()
        if not answer:
            return {"content": "", "sources": [], "reason": "answer_question_empty"}

        title = str(data.get("title") or "").strip()
        selected_pages = data.get("selected_pages") or []
        if not isinstance(selected_pages, list):
            selected_pages = [selected_pages]
        selected_pages = [str(page) for page in selected_pages if str(page).strip()]

        lines = [
            "[M3OS WIKI MEDICINAL CHEMISTRY KNOWLEDGE]",
            f"Question: {question}",
        ]
        if title:
            lines.append(f"Title: {title}")
        if selected_pages:
            lines.append("Selected pages:")
            lines.extend(f"- {page}" for page in selected_pages)
        lines.extend(["", "Answer:", answer])

        return {
            "content": "\n".join(lines).strip(),
            "sources": [
                {
                    "source_type": "m3os_wiki",
                    "tool": "answer_question",
                    "content": answer,
                    "metadata": {
                        "title": title,
                        "selected_pages": selected_pages,
                        "index_found": data.get("index_found"),
                    },
                    "score": None,
                }
            ],
            "metadata": {
                "title": title,
                "selected_pages": selected_pages,
                "index_found": data.get("index_found"),
            },
        }

    async def _search_and_read(self, query: str) -> Dict[str, Any]:
        if not (self._has_tool("search_articles") and self._has_tool("read_article")):
            return {"content": "", "sources": [], "reason": "search_or_read_unavailable"}

        try:
            raw_results = await self.mcp_client.invoke_tool(
                "search_articles",
                {"query": query, "limit": self.max_pages},
            )
        except Exception as exc:
            return {
                "content": "",
                "sources": [],
                "reason": f"search_articles_failed: {type(exc).__name__}: {exc}",
            }

        results = self._coerce_structured(raw_results)
        if isinstance(results, dict):
            results = results.get("results") or results.get("articles") or []
        if not isinstance(results, list) or not results:
            return {"content": "", "sources": [], "reason": "search_articles_empty"}

        sections: List[str] = [
            "[M3OS WIKI MEDICINAL CHEMISTRY KNOWLEDGE]",
            f"Search query: {query}",
            "",
        ]
        sources: List[Dict[str, Any]] = []

        for index, ref in enumerate(results[: self.max_pages], start=1):
            if not isinstance(ref, dict):
                continue
            name_or_id = str(ref.get("id") or ref.get("name") or "").strip()
            if not name_or_id:
                continue
            article = await self._read_article(name_or_id)
            if not article:
                continue
            body = self._trim(article.get("body", ""), self.article_max_chars)
            name = str(article.get("name") or ref.get("name") or name_or_id)
            frontmatter = article.get("frontmatter") if isinstance(article.get("frontmatter"), dict) else {}
            status = frontmatter.get("status") or ref.get("status")
            confidence = frontmatter.get("confidence") or ref.get("confidence")
            sections.append(
                "\n".join(
                    [
                        f"Article {index}: {name}",
                        f"ID: {article.get('id') or ref.get('id') or name_or_id}",
                        f"Status: {status or 'unknown'}",
                        f"Confidence: {confidence or 'unknown'}",
                        f"Score: {ref.get('score') if ref.get('score') is not None else 'unknown'}",
                        "Content:",
                        body,
                    ]
                ).strip()
            )
            sources.append(
                {
                    "source_type": "m3os_wiki",
                    "tool": "search_articles/read_article",
                    "content": body,
                    "metadata": {
                        "article_id": article.get("id") or ref.get("id"),
                        "name": name,
                        "path": article.get("path"),
                        "status": status,
                        "confidence": confidence,
                    },
                    "score": ref.get("score"),
                }
            )

        if not sources:
            return {"content": "", "sources": [], "reason": "read_article_empty"}

        return {
            "content": "\n\n".join(sections).strip(),
            "sources": sources,
            "metadata": {"query": query, "result_count": len(sources)},
        }

    async def _read_article(self, name_or_id: str) -> Optional[Dict[str, Any]]:
        try:
            raw = await self.mcp_client.invoke_tool("read_article", {"name_or_id": name_or_id})
        except Exception:
            return None
        data = self._coerce_structured(raw)
        return data if isinstance(data, dict) else None

    def _has_tool(self, tool_name: str) -> bool:
        has_tool = getattr(self.mcp_client, "has_tool", None)
        return bool(callable(has_tool) and has_tool(tool_name))

    @staticmethod
    def _build_question(query: str, task_context: Optional[str]) -> str:
        parts = [
            "Answer as a medicinal chemistry knowledge base for M3OS.",
            f"Knowledge need: {query}",
        ]
        if task_context:
            parts.append(f"Task context: {task_context}")
        parts.append(
            "Focus on structure-property relationships, ADMET mechanisms, "
            "bioisosteres, scaffold/risk alerts, and actionable design principles."
        )
        return "\n".join(parts)

    @classmethod
    def _coerce_structured(cls, value: Any) -> Any:
        if isinstance(value, tuple) and value:
            value = value[0]
        if isinstance(value, list):
            if not value:
                return value
            if all(cls._looks_like_text_content(item) for item in value):
                text = cls._stringify_result(value).strip()
                return cls._parse_text(text) if text else value
            if len(value) == 1 and cls._looks_like_text_content(value[0]):
                text = cls._stringify_result(value[0]).strip()
                return cls._parse_text(text) if text else value
            return value
        if isinstance(value, dict):
            return value
        text = cls._stringify_result(value).strip()
        return cls._parse_text(text) if text else value

    @staticmethod
    def _looks_like_text_content(value: Any) -> bool:
        if isinstance(value, str):
            return True
        if isinstance(value, dict):
            content_type = value.get("type")
            if "text" in value and content_type in {None, "text"}:
                return True
            if (
                "content" in value
                and content_type in {None, "text"}
                and len(value.keys()) <= 3
            ):
                return True
            return False
        return getattr(value, "text", None) is not None

    @staticmethod
    def _parse_text(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            return ast.literal_eval(text)
        except Exception:
            return text

    @classmethod
    def _stringify_result(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if "text" in value:
                return str(value.get("text") or "")
            if "content" in value:
                return cls._stringify_result(value.get("content"))
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, list):
            return "\n".join(cls._stringify_result(item) for item in value if item is not None)
        text = getattr(value, "text", None)
        if text is not None:
            return str(text)
        content = getattr(value, "content", None)
        if content is not None:
            return cls._stringify_result(content)
        return str(value)

    @staticmethod
    def _trim(value: Any, max_chars: int) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        omitted = len(text) - max_chars
        return f"{text[:max_chars].rstrip()}\n\n[TRUNCATED: omitted {omitted} characters]"
