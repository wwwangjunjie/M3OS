"""Base agent class for M3OS.

This module provides an abstract base class for all agents in the system,
encapsulating common functionality like LLM setup, middleware configuration,
and MCP client management.
"""

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional
import ast
import hashlib
import json
import time

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain_core.tools import StructuredTool

from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.core.retry_utils import (
    invoke_with_retry,
    validate_structured_response,
)
from m3os.agents_v5.core.trace_utils import build_invocation_trace, safe_serialize
from m3os.agents_v5.middleware.role_scoped_skills import RoleScopedSkillsMiddleware
from m3os.agents_v5.services.llm_factory import create_base_model, get_middleware
from m3os.agents_v5.services.mcp_client import MCPClientManager
from m3os.agents_v5.services.admet_property_selection import (
    enforce_frozen_admet_preference_json,
)

AgentEventSink = Callable[[Dict[str, Any]], Awaitable[None] | None]


def extract_todos_from_payload(value: Any) -> List[Dict[str, str]]:
    """Best-effort extraction of DeepAgents write_todos payloads."""
    return _extract_todos_from_payload(value, set())


def _extract_todos_from_payload(value: Any, seen: set[int]) -> List[Dict[str, str]]:
    if value is None:
        return []
    value_id = id(value)
    if value_id in seen:
        return []
    seen.add(value_id)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _extract_todos_from_payload(model_dump(), seen)
        except (AttributeError, TypeError):
            pass
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            return _extract_todos_from_payload(dict_method(), seen)
        except (AttributeError, TypeError):
            pass
    if not isinstance(value, (dict, list, str, bytes)):
        attrs = {
            key: getattr(value, key, None)
            for key in (
                "todos",
                "args",
                "input",
                "output",
                "messages",
                "message",
                "tool_calls",
                "invalid_tool_calls",
                "additional_kwargs",
                "kwargs",
                "data",
                "content",
            )
            if getattr(value, key, None) is not None
        }
        if attrs:
            return _extract_todos_from_payload(attrs, seen)
    if isinstance(value, dict):
        if isinstance(value.get("todos"), list):
            return _normalize_todo_items(value.get("todos"))
        update = value.get("update")
        if isinstance(update, dict) and isinstance(update.get("todos"), list):
            return _normalize_todo_items(update.get("todos"))
        for key in (
            "output",
            "input",
            "args",
            "messages",
            "message",
            "tool_calls",
            "invalid_tool_calls",
            "additional_kwargs",
            "kwargs",
            "data",
            "content",
        ):
            nested = _extract_todos_from_payload(value.get(key), seen)
            if nested:
                return nested
        return []
    if isinstance(value, list):
        if _looks_like_todo_list(value):
            return _normalize_todo_items(value)
        for item in reversed(value):
            nested = _extract_todos_from_payload(item, seen)
            if nested:
                return nested
        return []
    text = str(value or "").strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        nested = _extract_todos_from_payload(parsed, seen)
        if nested:
            return nested
    marker = "Updated todo list to "
    if marker in text:
        tail = text.split(marker, 1)[1].strip()
        try:
            return _normalize_todo_items(ast.literal_eval(tail))
        except Exception:
            return []
    return []


def _looks_like_todo_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    dict_items = [item for item in value if isinstance(item, dict)]
    if not dict_items:
        return False
    if len(dict_items) != len(value):
        return False
    if not any("status" in item for item in dict_items):
        return False
    return all(
        any(key in item for key in ("content", "task", "title"))
        for item in dict_items
    )


def extract_tool_call_id_from_payload(value: Any) -> str:
    """Best-effort extraction of a write_todos tool_call_id from event payloads."""
    if value is None:
        return ""
    direct = getattr(value, "tool_call_id", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    if isinstance(value, dict):
        direct = value.get("tool_call_id")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        for key in ("messages", "message", "update", "output", "input", "data", "kwargs"):
            nested = extract_tool_call_id_from_payload(value.get(key))
            if nested:
                return nested
        return ""
    if isinstance(value, list):
        for item in value:
            nested = extract_tool_call_id_from_payload(item)
            if nested:
                return nested
    return ""


def is_write_todos_tool_name(tool_name: Any) -> bool:
    normalized = str(tool_name or "").strip().lower()
    return normalized == "write_todos" or normalized.endswith(".write_todos")


def build_todo_signature(todos: Any) -> str:
    return json.dumps(_normalize_todo_items(todos), ensure_ascii=False, sort_keys=True)


def build_todo_update_payload(
    todos: Any,
    *,
    agent_type: str,
    agent_name: str,
    round_number: Any = None,
    round_index: Any = None,
    mcgs_round: Any = None,
    display_mcgs_round: Any = None,
    tool_call_id: str = "",
    todo_update_index: int = 0,
    todo_signature: str = "",
) -> Dict[str, Any]:
    normalized = _normalize_todo_items(todos)
    signature = todo_signature or build_todo_signature(normalized)
    signature_hash = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12] if signature else "empty"
    safe_agent = str(agent_type or agent_name or "m3os").strip().replace(" ", "_") or "m3os"
    event_token = str(tool_call_id or todo_update_index or signature_hash).strip()
    todo_event_id = f"{safe_agent}:{todo_update_index or 0}:{event_token}:{signature_hash}"
    total = len(normalized)
    completed = sum(1 for item in normalized if item.get("status") == "completed")
    in_progress = sum(1 for item in normalized if item.get("status") == "in_progress")
    pending = sum(1 for item in normalized if item.get("status") == "pending")
    return {
        "agent_type": agent_type,
        "agent_name": agent_name,
        "round": round_number,
        "round_index": round_index,
        "mcgs_round": mcgs_round,
        "display_mcgs_round": display_mcgs_round if display_mcgs_round is not None else mcgs_round,
        "todos": normalized,
        "total": total,
        "completed": completed,
        "in_progress": in_progress,
        "pending": pending,
        "progress": round(completed / total, 4) if total else 0,
        "tool_call_id": tool_call_id,
        "todo_signature": signature,
        "todo_update_index": todo_update_index,
        "todo_event_id": todo_event_id,
    }


def _normalize_todo_items(value: Any) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        return []
    todos: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or item.get("task") or item.get("title") or "").strip()
        status = str(item.get("status") or "pending").strip().lower()
        if status not in {"pending", "in_progress", "completed"}:
            status = "pending"
        if content:
            todos.append({"content": content, "status": status})
    return todos


class BaseAgent(ABC):
    """Abstract base class for molecular optimization agents.
    
    This class provides common infrastructure for all agents, including:
    - LLM model creation and configuration
    - Middleware setup (retry, caching)
    - MCP client integration
    - Agent lifecycle management
    """
    
    def __init__(
        self,
        config: AgentConfig,
        mcp_client: MCPClientManager,
        additional_context: Optional[str] = None,
        event_sink: Optional[AgentEventSink] = None,
        medchem_retrieval_agent: Optional[Any] = None,
    ):
        """Initialize the base agent.
        
        Args:
            config: Agent configuration
            mcp_client: MCP client manager for tool access
            additional_context: Optional user-provided context to inject at system prompt start
        """
        self.config = config
        self.mcp_client = mcp_client
        self.additional_context = additional_context or ""
        self.event_sink = event_sink
        self.medchem_retrieval_agent = medchem_retrieval_agent
        self.medchem_answer_memory: List[Dict[str, Any]] = []
        self.event_context: Dict[str, Any] = {}
        self._agent: Optional[Any] = None
        self._last_todo_signature = ""
        self._todo_update_index = 0
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Get the system prompt for this agent.
        
        Returns:
            The system prompt string with any formatting applied
        """
        pass
    
    @property
    @abstractmethod
    def output_schema(self) -> Optional[type]:
        """Get the Pydantic output schema for this agent.
        
        Returns:
            Pydantic BaseModel class for structured output, or None for free-form output.
        """
        pass

    @property
    @abstractmethod
    def agent_role(self) -> str:
        """Get the role key used for role-specific skill loading."""
        pass
    
    @abstractmethod
    async def get_tools(self) -> List[Any]:
        """Get the list of tools available to this agent.
        
        Returns:
            List of LangChain tools
        """
        pass
    
    def _create_model(self) -> Any:
        """Create the LLM model for this agent.
        
        Returns:
            Configured LangChain chat model
        """
        return create_base_model(self.config.llm)
    
    def _get_middleware(self) -> List[Any]:
        """Get middleware for this agent.
        
        Returns:
            List of middleware instances
        """
        return get_middleware(self.config.llm)

    @property
    def inject_shared_medchem_memory(self) -> bool:
        """Whether raw shared medchem tool memory should be prompt-injected."""
        return False
    
    async def build(self) -> Any:
        """Build and return the configured agent.
        
        This method creates the agent with all necessary configuration
        including LLM, system prompt, tools, and middleware.
        
        Returns:
            Configured LangChain agent
        """
        if self._agent is None:
            model = self._create_model()
            tools = [self._wrap_tool_with_trace(tool_item) for tool_item in await self.get_tools()]
            backend = FilesystemBackend(
                root_dir=self.config.deepagent.backend_root_dir,
                virtual_mode=True,
            )
            skill_sources = self.config.deepagent.get_skill_sources(self.agent_role)
            allowed_skill_names = self.config.deepagent.get_allowed_skill_names(self.agent_role)
            middleware = [
                RoleScopedSkillsMiddleware(
                    backend=backend,
                    sources=skill_sources,
                    allowed_skill_names=allowed_skill_names,
                ),
                *self._get_middleware(),
            ]

            agent_kwargs = {
                "model": model,
                "system_prompt": self.system_prompt,
                "tools": tools,
                "skills": None,
                "backend": backend,
                "middleware": middleware,
            }
            if self.output_schema is not None:
                agent_kwargs["response_format"] = self.output_schema

            self._agent = create_deep_agent(**agent_kwargs)
        
        return self._agent

    def set_event_sink(self, event_sink: Optional[AgentEventSink]) -> None:
        self.event_sink = event_sink

    def set_medchem_retrieval_agent(self, medchem_retrieval_agent: Optional[Any]) -> None:
        self.medchem_retrieval_agent = medchem_retrieval_agent

    def _build_ask_medchem_knowledge_tool(self) -> Any:
        async def ask_medchem_knowledge(
            question: str,
            decision_context: str = "",
            answer_focus: str = "",
        ) -> str:
            """Ask the dedicated medicinal-chemistry retrieval agent for concise guidance."""
            return await self.ask_medchem_knowledge(
                question=question,
                decision_context=decision_context,
                answer_focus=answer_focus,
            )

        return StructuredTool.from_function(
            coroutine=ask_medchem_knowledge,
            name="ask_medchem_knowledge",
            description=(
                "Ask the dedicated medicinal-chemistry retrieval agent for concise, "
                "source-aware guidance. Use this whenever SAR, ADMET mechanism, "
                "bioisostere, scaffold-liability, uploaded-paper/table evidence, "
                "or scoring/design risk is unclear. First reuse this agent's "
                "lightweight medchem answer memory when it already covers the "
                "decision; otherwise ask a focused question."
            ),
        )

    async def ask_medchem_knowledge(
        self,
        *,
        question: str,
        decision_context: str = "",
        answer_focus: str = "",
    ) -> str:
        question = str(question or "").strip()
        if not question:
            return json.dumps(
                {
                    "answer": "",
                    "key_points": [],
                    "sources": [],
                    "limitations": ["No medicinal-chemistry question was provided."],
                    "memory_hit": False,
                },
                ensure_ascii=False,
            )
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
        answer_method = getattr(self.medchem_retrieval_agent, "answer", None)
        if callable(answer_method):
            raw_answer = await answer_method(
                question=question,
                decision_context=decision_context,
                answer_focus=answer_focus,
                requester_role=self.agent_role,
                requester_context=dict(self.event_context),
            )
        else:
            raw_answer = await self.medchem_retrieval_agent.invoke(
                self._format_medchem_retrieval_prompt(
                    question=question,
                    decision_context=decision_context,
                    answer_focus=answer_focus,
                    requester_role=self.agent_role,
                )
            )
        answer_text = self._medchem_answer_to_text(raw_answer)
        answer_text = self._compact_answer_text(answer_text, 3000)
        self._remember_medchem_answer(question, answer_text)
        return answer_text

    def set_event_context(self, **context: Any) -> None:
        self.event_context = {
            key: value for key, value in context.items()
            if value is not None and value != ""
        }

    def _wrap_tool_with_trace(self, tool_item: Any) -> Any:
        tool_name = getattr(tool_item, "name", str(tool_item))
        description = self._tool_description_for_wrapping(
            tool_name,
            getattr(tool_item, "description", "") or "",
        )
        args_schema = getattr(tool_item, "args_schema", None)
        return_direct = getattr(tool_item, "return_direct", False)

        async def traced_tool(**kwargs: Any) -> Any:
            started_at = time.perf_counter()
            await self._emit_event(
                "m3os_subagent_tool_call",
                self._format_tool_call_event(tool_name),
            )
            try:
                result = await self._invoke_wrapped_tool(tool_item, tool_name, kwargs)
            except Exception as exc:
                elapsed = time.perf_counter() - started_at
                await self._emit_event(
                    "m3os_subagent_tool_summary",
                    self._format_tool_summary_event(
                        tool_name,
                        None,
                        elapsed=elapsed,
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
                if self.mcp_client.is_connection_error(exc):
                    return self._format_tool_failure_result(tool_name, exc)
                raise

            elapsed = time.perf_counter() - started_at
            await self._emit_event(
                "m3os_subagent_tool_summary",
                self._format_tool_summary_event(tool_name, result, elapsed=elapsed),
            )
            return result

        return StructuredTool.from_function(
            coroutine=traced_tool,
            name=tool_name,
            description=description,
            args_schema=args_schema,
            return_direct=return_direct,
        )

    def _tool_description_for_wrapping(self, tool_name: str, description: str) -> str:
        return description

    async def _invoke_wrapped_tool(self, tool_item: Any, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        if tool_name == "admet_predict_by_admetai":
            kwargs = dict(kwargs)
            kwargs["preference_json"] = enforce_frozen_admet_preference_json(
                kwargs.get("preference_json"),
                self.event_context.get("selected_admet_preference_json"),
                tool_name=tool_name,
            )
        if self.mcp_client.has_tool(tool_name):
            try:
                registered_tool = self.mcp_client.get_tool(tool_name)
            except ValueError:
                registered_tool = None
            if registered_tool is tool_item:
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
        label, stage, message = self._tool_metadata(tool_name)
        return {
            **self._event_context_payload(),
            "tool": tool_name,
            "label": label,
            "stage": stage,
            "message": message,
        }

    def _format_tool_summary_event(
        self,
        tool_name: str,
        result: Any,
        *,
        elapsed: float,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        label, stage, _ = self._tool_metadata(tool_name)
        safe_result = safe_serialize(result)
        molecules = self._extract_molecule_preview(safe_result)
        metrics = self._extract_tool_metrics(safe_result)
        payload = {
            **self._event_context_payload(),
            "tool": tool_name,
            "label": label,
            "stage": stage,
            "status": "failed" if error else "completed",
            "elapsed_seconds": round(elapsed, 2),
            "summary": error or self._summarize_tool_result(tool_name, safe_result, molecules, metrics),
            "molecules": molecules,
            "metrics": metrics,
            "display": self._format_tool_display(tool_name, safe_result),
        }
        return payload

    async def _emit_event(self, event_type: str, content: Dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        payload = {
            "type": event_type,
            "source": self._event_source_label(),
            "content": content,
        }
        result = self.event_sink(payload)
        if hasattr(result, "__await__"):
            await result

    def _event_context_payload(self) -> Dict[str, Any]:
        round_value = self.event_context.get("round")
        display_round = self.event_context.get("display_round") or round_value
        mcgs_round = self.event_context.get("mcgs_round") or display_round
        display_mcgs_round = self.event_context.get("display_mcgs_round") or mcgs_round
        return {
            "agent_type": self.event_context.get("agent_type") or self.agent_role,
            "round": round_value,
            "display_round": display_round,
            "round_index": self.event_context.get("round_index"),
            "mcgs_round": mcgs_round,
            "display_mcgs_round": display_mcgs_round,
        }

    def _event_source_label(self) -> str:
        role = self.agent_role
        if role == "creative":
            return "Creative Explorer"
        if role == "rational":
            return "Rational Designer"
        if role == "critic":
            return "Critic"
        return role.title()

    def _tool_metadata(self, tool_name: str) -> tuple[str, str, str]:
        lower = tool_name.lower()
        if "validate_smiles" in lower or ("smiles" in lower and "valid" in lower):
            return "SMILES validation", "validation", "Checking generated SMILES with RDKit."
        if "rgroup" in lower or "r_group" in lower:
            return "R-group generation", "generation", "Generating scaffold decorations."
        if "similar_molecule" in lower or "mol2mol" in lower:
            return "Similar molecule generation", "generation", "Generating whole-molecule similar variants."
        if "linker" in lower or "linkinvent" in lower:
            return "Linker generation", "generation", "Generating linkers between molecular fragments."
        if "admet_filter" in lower:
            return "ADMET filtering", "evaluation", "Filtering generated molecules by ADMET preferences."
        if "iupac" in lower:
            return "IUPAC naming", "identity", "Generating molecule names."
        if "fragment" in lower or "functional" in lower:
            return "Fragment analysis", "analysis", "Identifying functional fragments."
        if "admet" in lower or "property" in lower or "predict" in lower:
            return "Property prediction", "evaluation", "Predicting molecular properties."
        if "reinvent" in lower or "generate" in lower or "sample" in lower:
            return "Molecule generation", "generation", "Generating candidate molecules."
        if "rag" in lower or "knowledge" in lower or "retrieve" in lower or "case" in lower:
            return "Knowledge retrieval", "knowledge", "Retrieving relevant chemistry evidence."
        return tool_name.replace("_", " ").title(), "tool", f"Running {tool_name}."

    def _extract_tool_metrics(self, value: Any) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if isinstance(value, dict):
            for key, item in value.items():
                if len(metrics) >= 6:
                    break
                if isinstance(item, (str, int, float, bool)) or item is None:
                    text = str(item)
                    metrics[str(key)] = text if len(text) <= 80 else f"{len(text)} chars"
                elif isinstance(item, list):
                    metrics[str(key)] = len(item)
                elif isinstance(item, dict):
                    metrics[str(key)] = len(item)
        elif isinstance(value, list):
            metrics["items"] = len(value)
        elif isinstance(value, str):
            metrics["chars"] = len(value)
        return metrics

    def _extract_molecule_preview(self, value: Any, limit: int = 4) -> List[Dict[str, Any]]:
        molecules: List[Dict[str, Any]] = []

        def visit(item: Any) -> None:
            if len(molecules) >= limit:
                return
            if isinstance(item, dict):
                smiles = item.get("smiles") or item.get("SMILES")
                if smiles:
                    molecules.append({
                        "smiles": str(smiles),
                        "label": str(item.get("label") or item.get("name") or "Tool molecule"),
                        "score": item.get("score"),
                        "properties": item.get("properties") if isinstance(item.get("properties"), dict) else {},
                    })
                    return
                for child in item.values():
                    visit(child)
                    if len(molecules) >= limit:
                        return
            elif isinstance(item, list):
                for child in item:
                    visit(child)
                    if len(molecules) >= limit:
                        return

        visit(value)
        return molecules

    def _format_tool_display(self, tool_name: str, value: Any) -> Dict[str, Any]:
        lower = tool_name.lower()
        if isinstance(value, dict):
            if "iupac" in lower:
                return {"iupac": self._compact_preview(value)}
            if "fragment" in lower:
                return {"fragments": self._compact_preview(value)}
        return {"preview": self._compact_preview(value)}

    def _summarize_tool_result(
        self,
        tool_name: str,
        value: Any,
        molecules: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> str:
        label, _, _ = self._tool_metadata(tool_name)
        if molecules:
            return f"{label} completed with {len(molecules)} molecule preview{'s' if len(molecules) != 1 else ''}."
        if metrics:
            metric_text = ", ".join(f"{key}: {val}" for key, val in list(metrics.items())[:3])
            return f"{label} completed ({metric_text})."
        return f"{label} completed."

    def _compact_preview(self, value: Any, max_chars: int = 420) -> str:
        if isinstance(value, str):
            text = value
        else:
            text = str(value)
        text = " ".join(text.split())
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars].rstrip()}..."
    
    def update_additional_context(self, additional_context: str) -> None:
        """Update the additional user context.
        
        This method allows updating context after agent initialization.
        
        Args:
            additional_context: New additional context
        """
        self.additional_context = additional_context
        # Rebuild agent to use new context
        self._agent = None
    
    async def invoke(self, messages: Any) -> Dict[str, Any]:
        """Invoke the agent with messages.

        Args:
            messages: Messages to send to the agent. Can be a string (will be
                     wrapped in HumanMessage) or a list of message objects.

        Returns:
            Agent response dictionary
        """
        agent = await self.build()
        normalized_messages = self._normalize_messages(messages)
        self._last_todo_signature = ""
        self._todo_update_index = 0
        await self._emit_event(
            "m3os_agent_message",
            {
                **self._event_context_payload(),
                "agent_name": self._event_source_label(),
                "stage": "reasoning",
                "message": self._agent_start_message(),
            },
        )

        async def _invoke():
            result = await self._invoke_agent_with_todo_events(agent, normalized_messages)
            if self.output_schema is None and isinstance(result, dict):
                result["raw_response_text"] = self._extract_response_text(result)
                if not result["raw_response_text"] and result.get("streaming_partial_state"):
                    result["raw_response_text"] = self._streaming_partial_response_text(result)
            return result
        
        start = time.perf_counter()
        validate_fn = (
            validate_structured_response
            if self.output_schema is not None
            else self._validate_text_response
        )
        result = await invoke_with_retry(
            _invoke,
            max_retries=3,
            base_delay=1.0,
            validate_fn=validate_fn,
            operation_name=f"{self.__class__.__name__} agent invocation"
        )

        elapsed = round(time.perf_counter() - start, 3)
        llm_usage = self._extract_usage_metadata(result)
        result["llm_usage"] = {
            "elapsed_sec": elapsed,
            "usage": llm_usage,
        }
        result["invocation_trace"] = build_invocation_trace(
            agent_name=self.__class__.__name__,
            input_messages=normalized_messages,
            result=result,
        )
        await self._emit_event(
            "m3os_agent_message",
            {
                **self._event_context_payload(),
                "agent_name": self._event_source_label(),
                "stage": "summary",
                "message": self._agent_result_message(result),
            },
        )
        print(f"[{self.__class__.__name__} usage]: {result['llm_usage']}")
        return result

    async def finalize_audit_summary(
        self,
        result: Any,
        *,
        audit_kind: str,
        source_agent_type: str,
        extra_instruction: str = "",
    ) -> Dict[str, Any]:
        """Produce the final auditor-facing summary after all tool use is done.

        This is intentionally a direct base-model call, not a DeepAgent call, so
        no tools are available and the model has exactly one job: emit the
        structured Audit-friendly summary that the Auditor will consume.
        """
        finalization_prompt = self._audit_finalization_prompt(
            audit_kind=audit_kind,
            source_agent_type=source_agent_type,
            extra_instruction=extra_instruction,
        )
        transcript_messages = self._audit_finalization_transcript_messages(result, finalization_prompt)
        use_transcript_input = bool(isinstance(result, dict) and result.get("streaming_partial_state"))
        direct_messages = (
            transcript_messages
            if use_transcript_input
            else self._audit_finalization_messages(result, finalization_prompt)
        )
        model = self._create_model()
        warnings: List[str] = []
        used_transcript_fallback = False
        direct_error = ""

        async def _invoke_direct() -> Any:
            return await model.ainvoke(direct_messages)

        async def _invoke_transcript() -> Any:
            return await model.ainvoke(transcript_messages)

        start = time.perf_counter()
        try:
            raw_result = await invoke_with_retry(
                _invoke_direct,
                max_retries=1,
                base_delay=1.0,
                validate_fn=self._validate_finalization_text_response,
                operation_name=f"{self.__class__.__name__} audit finalization",
            )
        except Exception as exc:
            direct_error = f"{type(exc).__name__}: {exc}"
            warnings.append(
                "Direct conversation replay for audit finalization failed; used plain-text transcript fallback."
            )
            used_transcript_fallback = True
            raw_result = await invoke_with_retry(
                _invoke_transcript,
                max_retries=3,
                base_delay=1.0,
                validate_fn=self._validate_finalization_text_response,
                operation_name=f"{self.__class__.__name__} audit finalization transcript fallback",
            )

        elapsed = round(time.perf_counter() - start, 3)
        source_text = self._extract_model_response_text(raw_result)
        usage = self._extract_model_usage_metadata(raw_result)
        return {
            "source_text": source_text,
            "source_warnings": warnings,
            "finalization_trace": {
                "audit_kind": audit_kind,
                "source_agent_type": source_agent_type,
                "elapsed_sec": elapsed,
                "usage": safe_serialize(usage),
                "used_transcript_input": use_transcript_input,
                "used_transcript_fallback": used_transcript_fallback,
                "direct_error": direct_error,
                "summary_chars": len(source_text),
                "prompt": finalization_prompt,
            },
            "llm_usage": {
                "elapsed_sec": elapsed,
                "usage": usage,
            },
        }

    async def _invoke_agent_with_todo_events(self, agent: Any, normalized_messages: Any) -> Dict[str, Any]:
        astream_events = getattr(agent, "astream_events", None)
        if callable(astream_events):
            try:
                return await self._invoke_agent_streaming(agent, normalized_messages)
            except RuntimeError as exc:
                if str(exc) != "agent-streaming-unavailable":
                    raise
        result = await agent.ainvoke({"messages": normalized_messages})
        await self._emit_todo_update_from_result(result)
        return result

    async def _invoke_agent_streaming(self, agent: Any, normalized_messages: Any) -> Dict[str, Any]:
        final_result: Any = None
        latest_agent_state: Any = None
        saw_event = False
        try:
            async for event in agent.astream_events(
                {"messages": normalized_messages},
                version="v2",
            ):
                saw_event = True
                if not isinstance(event, dict):
                    continue
                event_name = str(event.get("event") or "")
                data = event.get("data") if isinstance(event.get("data"), dict) else {}
                if event_name in {"on_tool_start", "on_tool_end"}:
                    await self._emit_write_todos_update(event, data)
                if event_name in {"on_chain_end", "on_graph_end"} and data.get("output") is not None:
                    candidate = data.get("output")
                    if self._looks_like_agent_state_output(candidate):
                        latest_agent_state = candidate
                    if self._looks_like_agent_final_output(candidate):
                        final_result = candidate
        except Exception as exc:
            if self._is_empty_stream_generation_error(exc) and latest_agent_state is not None:
                return self._streaming_partial_state_result(latest_agent_state, exc)
            if not saw_event:
                raise RuntimeError("agent-streaming-unavailable") from exc
            raise

        if not saw_event or final_result is None:
            if latest_agent_state is not None:
                return self._streaming_partial_state_result(
                    latest_agent_state,
                    RuntimeError("Streaming ended without a final assistant response."),
                )
            raise RuntimeError("agent-streaming-unavailable")
        if not isinstance(final_result, dict):
            return {"output": final_result}
        await self._emit_todo_update_from_result(final_result)
        return final_result

    def _looks_like_agent_state_output(self, value: Any) -> bool:
        if not isinstance(value, dict) or self._looks_like_llm_result_payload(value):
            return False
        messages = value.get("messages")
        return isinstance(messages, list) and bool(messages)

    def _looks_like_agent_final_output(self, value: Any) -> bool:
        """Return True for DeepAgent final state, False for nested model artifacts."""
        if value is None or self._looks_like_llm_result_payload(value):
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict):
            messages = value.get("messages")
            if isinstance(messages, list) and messages:
                return bool(self._extract_response_text({"messages": messages}))
            if value.get("structured_response") is not None:
                return True
            for key in ("output", "response", "content", "text"):
                item = value.get(key)
                if isinstance(item, str) and item.strip():
                    return True
            return False
        content = getattr(value, "content", None)
        if isinstance(content, str) and content.strip():
            return True
        return False

    @staticmethod
    def _looks_like_llm_result_payload(value: Any) -> bool:
        if value is None:
            return False
        if value.__class__.__name__ == "LLMResult":
            return True
        if not isinstance(value, dict):
            return False
        payload_type = str(value.get("type") or "").strip()
        if payload_type == "LLMResult":
            return True
        return "generations" in value and ("llm_output" in value or "run" in value)

    @staticmethod
    def _is_empty_stream_generation_error(exc: BaseException) -> bool:
        return "No generations found in stream." in str(exc)

    def _streaming_partial_state_result(self, state: Any, exc: BaseException) -> Dict[str, Any]:
        if isinstance(state, dict):
            result = dict(state)
        else:
            result = {"output": state}
        result["streaming_partial_state"] = True
        result["streaming_error"] = f"{type(exc).__name__}: {exc}"
        return result

    @staticmethod
    def _streaming_partial_response_text(result: Dict[str, Any]) -> str:
        error = str(result.get("streaming_error") or "").strip()
        if error:
            return (
                "Streaming ended after tool use before a final assistant response. "
                f"The workflow will finalize from the captured message/tool transcript. Error: {error}"
            )
        return "Streaming ended after tool use before a final assistant response."

    async def _emit_write_todos_update(self, event: Dict[str, Any], data: Dict[str, Any]) -> None:
        tool_name = str(event.get("name") or data.get("name") or "")
        if not is_write_todos_tool_name(tool_name):
            return
        todos = (
            extract_todos_from_payload(data.get("input"))
            or extract_todos_from_payload(data.get("output"))
            or extract_todos_from_payload(data)
        )
        if todos:
            tool_call_id = (
                extract_tool_call_id_from_payload(data)
                or extract_tool_call_id_from_payload(event)
            )
            await self._emit_todo_update(todos, tool_call_id=tool_call_id)

    async def _emit_todo_update_from_result(self, result: Any) -> None:
        if isinstance(result, dict):
            todos = extract_todos_from_payload(result.get("todos")) or extract_todos_from_payload(result)
            if todos:
                await self._emit_todo_update(todos)

    async def _emit_todo_update(self, todos: Any, *, tool_call_id: str = "") -> None:
        signature = build_todo_signature(todos)
        if signature and signature == self._last_todo_signature:
            return
        self._last_todo_signature = signature
        self._todo_update_index += 1
        payload = build_todo_update_payload(
            todos,
            agent_type=self.event_context.get("agent_type") or self.agent_role,
            agent_name=self._event_source_label(),
            round_number=self.event_context.get("display_round") or self.event_context.get("round"),
            round_index=self.event_context.get("round_index"),
            mcgs_round=self.event_context.get("mcgs_round")
            or self.event_context.get("display_round")
            or self.event_context.get("round"),
            display_mcgs_round=self.event_context.get("display_mcgs_round")
            or self.event_context.get("mcgs_round")
            or self.event_context.get("display_round")
            or self.event_context.get("round"),
            tool_call_id=tool_call_id,
            todo_update_index=self._todo_update_index,
            todo_signature=signature,
        )
        await self._emit_event("m3os_todo_update", payload)

    def _agent_start_message(self) -> str:
        role = self.agent_role
        if role == "creative":
            return "Creative Explorer is exploring broader chemical modifications and generation strategies."
        if role == "rational":
            return "Rational Designer is integrating medicinal chemistry knowledge, cases, and design strategy."
        if role == "critic":
            return "Critic is evaluating candidate molecules against the optimization objective."
        return f"{self._event_source_label()} is analyzing the current task."

    def _agent_result_message(self, result: Dict[str, Any]) -> str:
        structured = result.get("structured_response") if isinstance(result, dict) else None
        optimizations = getattr(structured, "optimizations", None)
        count = len(optimizations) if isinstance(optimizations, list) else 0
        if self.output_schema is None:
            text = result.get("raw_response_text") if isinstance(result, dict) else ""
            if text:
                return f"{self._event_source_label()} completed a free-form reasoning response."
            return f"{self._event_source_label()} completed its reasoning step."
        if self.agent_role == "critic":
            return f"Critic completed evaluation for {count} candidate molecule{'s' if count != 1 else ''}."
        if self.agent_role in {"creative", "rational"}:
            return f"{self._event_source_label()} proposed {count} candidate molecule{'s' if count != 1 else ''}."
        return f"{self._event_source_label()} completed its reasoning step."

    def _validate_text_response(self, result: Any) -> bool:
        if not isinstance(result, dict):
            return bool(str(result or "").strip())
        return bool(str(result.get("raw_response_text") or self._extract_response_text(result)).strip())

    def _validate_finalization_text_response(self, result: Any) -> bool:
        return bool(self._extract_model_response_text(result).strip())

    def _audit_finalization_prompt(
        self,
        *,
        audit_kind: str,
        source_agent_type: str,
        extra_instruction: str = "",
    ) -> str:
        common = (
            "Now produce ONLY the Audit-friendly summary in the required schema.\n"
            "Do not call tools.\n"
            "Do not add prose outside the schema.\n"
            "Use only molecules, facts, scores, properties, rationales, and confidence values that are already present in your preceding analysis, final answer, or tool results.\n"
            "Do not introduce any new molecule, SMILES, property value, score, rationale, or claim.\n"
        )
        if extra_instruction:
            common += f"{extra_instruction.strip()}\n"

        if audit_kind == "molecule_optimizations":
            return (
                common
                + f"Source agent type: {source_agent_type}\n\n"
                "Required schema: a Markdown table with exactly these columns in this order:\n"
                "| smiles | modification_type | rationale | confidence_score |\n"
                "| --- | --- | --- | --- |\n"
                "Rules:\n"
                "- Include one row for each final proposed current-round molecule, capped at 7 rows.\n"
                "- Do not add rows to satisfy any user-requested total candidate count; MCGS accumulates candidates across rounds.\n"
                "- `modification_type` must be the concrete chemical operation made to the molecule.\n"
                "- `confidence_score` must be a number from 0.0 to 1.0; use 0.0 only if no confidence was stated.\n"
            )

        if audit_kind == "critic_evaluations":
            return (
                common
                + f"Source agent type: {source_agent_type}\n\n"
                "Required schema: a Markdown table with exactly these columns in this order:\n"
                "| smiles | action | rationale | score | properties | agent_type |\n"
                "| --- | --- | --- | --- | --- | --- |\n"
                "Rules:\n"
                "- Include one row for each evaluated molecule.\n"
                "- `action` must be the concrete chemical operation made to the molecule; it is not an accept/reject decision.\n"
                "- `score` must be a number from 0.0 to 1.0; use 0.0 only if no score was stated.\n"
                "- `properties` must be a compact JSON object such as {\"BBB_Martins\": 0.91}; use {} only if no properties were stated.\n"
                "- `agent_type` means the generator that created the candidate: `creative_molecule_explorer` or `rational_medicinal_designer`. Use an empty string if generator provenance was not explicitly stated; never use `critic` or `auditor`.\n"
            )

        return (
            common
            + f"Source agent type: {source_agent_type}\n\n"
            "Required schema: a compact Markdown table containing the auditor-required fields from the preceding work.\n"
        )

    def _audit_finalization_messages(self, result: Any, finalization_prompt: str) -> List[Any]:
        messages = result.get("messages") if isinstance(result, dict) else None
        if isinstance(messages, list) and messages:
            return [
                {"role": "system", "content": self.system_prompt},
                *messages,
                {"role": "user", "content": finalization_prompt},
            ]
        return self._audit_finalization_transcript_messages(result, finalization_prompt)

    def _audit_finalization_transcript_messages(self, result: Any, finalization_prompt: str) -> List[Dict[str, str]]:
        transcript = self._audit_finalization_transcript(result)
        user_prompt = (
            "You are finalizing the preceding agent work for Auditor extraction.\n\n"
            "[PRECEDING AGENT WORK TRANSCRIPT]\n"
            f"{transcript}\n\n"
            "[FINALIZATION INSTRUCTION]\n"
            f"{finalization_prompt}"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _audit_finalization_transcript(self, result: Any) -> str:
        if isinstance(result, str):
            return result.strip()
        if not isinstance(result, dict):
            return str(result or "").strip()

        parts: List[str] = []
        raw_text = result.get("raw_response_text")
        if isinstance(raw_text, str) and raw_text.strip():
            parts.append(f"[raw_response_text]\n{raw_text.strip()}")

        messages = result.get("messages")
        if isinstance(messages, list):
            for index, message in enumerate(messages):
                role = self._message_role_for_transcript(message)
                content = self._message_content_for_transcript(message)
                if not content:
                    continue
                parts.append(f"[message {index}: {role}]\n{self._truncate_for_transcript(content)}")

        if not parts:
            parts.append(json.dumps(safe_serialize(result), ensure_ascii=False, default=str))

        transcript = "\n\n".join(parts).strip()
        return self._truncate_for_transcript(transcript, max_chars=80000)

    def _message_role_for_transcript(self, message: Any) -> str:
        if isinstance(message, dict):
            return str(message.get("role") or message.get("type") or message.get("name") or "message")
        return str(
            getattr(message, "role", None)
            or getattr(message, "type", None)
            or getattr(message, "name", None)
            or message.__class__.__name__
        )

    def _message_content_for_transcript(self, message: Any) -> str:
        if isinstance(message, dict):
            content = self._message_content_to_text(message.get("content"))
            tool_calls = message.get("tool_calls") or message.get("invalid_tool_calls")
        else:
            content = self._message_content_to_text(getattr(message, "content", None))
            tool_calls = getattr(message, "tool_calls", None) or getattr(message, "invalid_tool_calls", None)

        if tool_calls:
            tool_text = json.dumps(safe_serialize(tool_calls), ensure_ascii=False, default=str)
            content = f"{content}\n[tool_calls]\n{tool_text}".strip()
        return content.strip()

    @staticmethod
    def _truncate_for_transcript(text: str, max_chars: int = 12000) -> str:
        if len(text) <= max_chars:
            return text
        head = max_chars // 2
        tail = max_chars - head
        return f"{text[:head].rstrip()}\n\n[... truncated for audit finalization ...]\n\n{text[-tail:].lstrip()}"

    def _extract_response_text(self, result: Dict[str, Any]) -> str:
        """Best-effort extraction of the final natural-language agent response."""
        direct_keys = ("output", "response", "content", "text")
        for key in direct_keys:
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        messages = result.get("messages") if isinstance(result, dict) else None
        if not messages:
            return ""

        for msg in reversed(messages):
            if isinstance(msg, dict):
                message_type = str(msg.get("type") or "").lower()
                role = str(msg.get("role") or "").lower()
                tool_call_id = msg.get("tool_call_id")
                content = msg.get("content")
            else:
                message_type = str(getattr(msg, "type", "") or "").lower()
                role = str(getattr(msg, "role", "") or "").lower()
                tool_call_id = getattr(msg, "tool_call_id", None)
                content = getattr(msg, "content", None)
            if message_type == "tool" or role == "tool" or tool_call_id is not None:
                continue
            if role and role not in {"assistant", "ai"}:
                continue
            if message_type and message_type not in {"assistant", "ai", "aimessage", "simplenamespace"}:
                continue
            text = self._message_content_to_text(content)
            if text:
                return text
        return ""

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
        return ""

    def _normalize_messages(self, messages: Any) -> Any:
        """Normalize plain text input into chat-message format."""
        memory_block = self._prompt_memory_block()
        if isinstance(messages, str):
            content = self._append_medchem_memory(messages, memory_block)
            return [{"role": "user", "content": content}]
        if memory_block and isinstance(messages, list):
            normalized = list(messages)
            for index in range(len(normalized) - 1, -1, -1):
                item = normalized[index]
                if not isinstance(item, dict) or item.get("role") != "user":
                    continue
                updated = dict(item)
                updated["content"] = self._append_medchem_memory(
                    updated.get("content"),
                    memory_block,
                )
                normalized[index] = updated
                return normalized
        return messages

    def _prompt_memory_block(self) -> str:
        sections = []
        if self.inject_shared_medchem_memory:
            shared_memory = self._shared_medchem_memory_block()
            if shared_memory:
                sections.append(shared_memory)
        answer_memory = self._medchem_answer_memory_block()
        if answer_memory:
            sections.append(answer_memory)
        return "\n\n".join(sections).strip()

    def _shared_medchem_memory_block(self) -> str:
        formatter = getattr(self.mcp_client, "format_medchem_memory", None)
        if not callable(formatter):
            return ""
        try:
            return str(formatter() or "").strip()
        except Exception:
            return ""

    def _medchem_answer_memory_block(self, *, max_items: int = 8, max_chars: int = 6000) -> str:
        if not self.medchem_answer_memory:
            return ""
        recent = self.medchem_answer_memory[-max(1, int(max_items)) :]
        text = json.dumps(recent, ensure_ascii=False, indent=2, default=str)
        if len(text) > max_chars:
            text = text[-max_chars:]
        return (
            "[LIGHTWEIGHT MEDICINAL CHEMISTRY ANSWER MEMORY]\n"
            "Reuse these concise answers from ask_medchem_knowledge before asking again. "
            "If they do not cover the current scaffold, property, risk, or decision, "
            "ask ask_medchem_knowledge a focused follow-up question.\n"
            f"{text}"
        )

    def _remember_medchem_answer(self, question: str, answer_text: str) -> None:
        parsed = self._parse_medchem_answer(answer_text)
        sources = parsed.get("sources") if isinstance(parsed.get("sources"), list) else []
        memory_key = json.dumps(
            {
                "question": str(question or "").strip(),
                "answer": parsed.get("answer") or answer_text,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        self.medchem_answer_memory = [
            item for item in self.medchem_answer_memory if item.get("key") != memory_key
        ]
        self.medchem_answer_memory.append(
            {
                "key": memory_key,
                "memory_type": "medchem_answer",
                "round_index": self.event_context.get("round_index"),
                "source_agent": "medchem_retrieval",
                "question": str(question or "").strip(),
                "answer": self._compact_answer_text(parsed.get("answer") or answer_text, 1600),
                "sources": sources[:8],
            }
        )
        self.medchem_answer_memory = self.medchem_answer_memory[-24:]

    @staticmethod
    def _parse_medchem_answer(answer_text: str) -> Dict[str, Any]:
        text = str(answer_text or "").strip()
        if not text:
            return {}
        candidates = [text]
        if text.startswith("```"):
            stripped = text.strip("`")
            if "\n" in stripped:
                candidates.append(stripped.split("\n", 1)[1].strip())
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def _medchem_answer_to_text(raw_answer: Any) -> str:
        if isinstance(raw_answer, str):
            return raw_answer.strip()
        if isinstance(raw_answer, dict):
            if isinstance(raw_answer.get("raw_response_text"), str):
                return raw_answer["raw_response_text"].strip()
            for key in ("answer", "content", "text", "output"):
                value = raw_answer.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            try:
                return json.dumps(raw_answer, ensure_ascii=False, default=str)
            except Exception:
                return str(raw_answer)
        content = getattr(raw_answer, "content", None)
        if isinstance(content, str):
            return content.strip()
        return str(raw_answer or "").strip()

    @staticmethod
    def _compact_answer_text(text: Any, max_chars: int) -> str:
        compact = str(text or "").strip()
        if len(compact) <= max_chars:
            return compact
        return compact[:max_chars].rstrip() + "\n...[medchem answer shortened]"

    @staticmethod
    def _format_medchem_retrieval_prompt(
        *,
        question: str,
        decision_context: str,
        answer_focus: str,
        requester_role: str,
    ) -> str:
        return (
            "[MEDCHEM KNOWLEDGE REQUEST]\n"
            f"Requester role: {requester_role}\n"
            f"Question: {question}\n"
            f"Decision context: {decision_context or 'N/A'}\n"
            f"Answer focus: {answer_focus or 'Concise design/scoring implications'}\n"
            "Return compact JSON with keys: answer, key_points, sources, limitations, memory_hit."
        )

    @staticmethod
    def _append_medchem_memory(content: Any, memory_block: str) -> Any:
        if not memory_block:
            return content
        if isinstance(content, str):
            if "[SHARED MEDICINAL CHEMISTRY KNOWLEDGE MEMORY]" in content:
                return content
            return f"{content.rstrip()}\n\n{memory_block}"
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str) and "[SHARED MEDICINAL CHEMISTRY KNOWLEDGE MEMORY]" in text:
                        return content
            return [
                *content,
                {"type": "text", "text": memory_block},
            ]
        return content

    def _extract_usage_metadata(self, result: Dict[str, Any]) -> Any:
        """Best-effort extraction of usage metadata from agent output."""
        messages = result.get("messages") if isinstance(result, dict) else None
        if not messages:
            return None

        for msg in reversed(messages):
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                return usage
        return None

    def _extract_model_response_text(self, result: Any) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            for key in ("raw_response_text", "output", "response", "content", "text"):
                value = result.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            messages = result.get("messages")
            if isinstance(messages, list):
                for message in reversed(messages):
                    if isinstance(message, dict):
                        content = message.get("content")
                    else:
                        content = getattr(message, "content", None)
                    text = self._message_content_to_text(content)
                    if text:
                        return text
            raw = result.get("raw")
            if raw is not None:
                return self._extract_model_response_text(raw)
            return ""

        content = getattr(result, "content", None)
        text = self._message_content_to_text(content)
        if text:
            return text
        return str(result or "").strip()

    @staticmethod
    def _extract_model_usage_metadata(result: Any) -> Any:
        usage = getattr(result, "usage_metadata", None)
        if usage:
            return usage
        response_metadata = getattr(result, "response_metadata", None)
        if isinstance(response_metadata, dict):
            return response_metadata.get("usage") or response_metadata.get("token_usage")
        if isinstance(result, dict):
            raw = result.get("raw")
            if raw is not None:
                return BaseAgent._extract_model_usage_metadata(raw)
            messages = result.get("messages")
            if isinstance(messages, list):
                for message in reversed(messages):
                    usage = getattr(message, "usage_metadata", None)
                    if usage:
                        return usage
        return None
