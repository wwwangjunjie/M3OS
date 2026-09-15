"""Node: Analyze Current Molecule (Shared Context)

This node performs one compact analysis for the current molecule and shares it
with Rational/Creative/Critic to reduce duplicated reasoning and token usage.
"""

import json
import time
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from m3os.agents_v5.core.state import MCGSState
from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.core.retry_utils import (
    invoke_with_retry,
)
from m3os.agents_v5.core.models import SharedMoleculeAnalysis
from m3os.agents_v5.services.llm_factory import create_base_model
from m3os.agents_v5.services.molecule_visual_context import (
    build_topology_visual_context,
    text_block,
)
from m3os.agents_v5.services.molecular_context import (
    without_explicit_molecular_auxiliary_fields,
)
from m3os.agents_v5.services.smiles_validation import validate_smiles
from m3os.agents_v5.prompts import (
    INITIAL_ANALYZE_SYSTEM_PROMPT,
    INITIAL_ANALYZE_USER_PROMPT,
)


class AnalyzeCurrentNode:
    """Build and cache shared analysis package for current molecule."""

    def __init__(self, config: AgentConfig):
        self.config = config

    async def __call__(self, state: MCGSState) -> Dict[str, Any]:
        print("\n=== Node: analyze_current ===")
        start = time.perf_counter()

        if not self.config.enable_shared_analysis:
            print("Shared analysis disabled by config.")
            return {
                "current_shared_analysis": self._empty_shared_analysis(),
                "runtime_metrics": [
                    {
                        "node": "analyze_current",
                        "enabled": False,
                        "elapsed_sec": round(time.perf_counter() - start, 3),
                    }
                ],
            }

        current_info = state["current_info_list"][-1] if state["current_info_list"] else {}
        current_smiles = current_info.get("current_smiles")
        current_key = self._canonical_smiles_key(current_smiles)
        cache = dict(state.get("shared_analysis_cache") or {})

        cached_analysis = cache.get(current_key) if current_key else None
        if cached_analysis is None and current_smiles:
            cached_analysis = cache.get(str(current_smiles))
        if current_key and cached_analysis is not None:
            print("Shared analysis cache hit.")
            cache[current_key] = cached_analysis
            return {
                "current_shared_analysis": cached_analysis,
                "shared_analysis_cache": cache,
                "runtime_metrics": [
                    {
                        "node": "analyze_current",
                        "cache_hit": True,
                        "elapsed_sec": round(time.perf_counter() - start, 3),
                    }
                ],
            }

        try:
            analysis, usage = await self._analyze(state, current_info)
            if current_key:
                cache[current_key] = analysis
            elapsed = round(time.perf_counter() - start, 3)

            print("[analyze_current usage]:", usage)
            print("\n===[END] Node: analyze_current ===")
            return {
                "current_shared_analysis": analysis,
                "shared_analysis_cache": cache,
                "runtime_metrics": [
                    {
                        "node": "analyze_current",
                        "cache_hit": False,
                        "elapsed_sec": elapsed,
                        "usage": usage,
                    }
                ],
            }
        except Exception as exc:
            print(f"Shared analysis failed, fallback to empty context: {exc}")
            return {
                "current_shared_analysis": self._empty_shared_analysis(),
                "shared_analysis_cache": cache,
                "runtime_metrics": [
                    {
                        "node": "analyze_current",
                        "fallback": True,
                        "error": str(exc),
                        "elapsed_sec": round(time.perf_counter() - start, 3),
                    }
                ],
            }

    async def _analyze(
        self,
        state: MCGSState,
        current_info: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Any]:
        llm = create_base_model(self.config.llm)
        try:
            structured_llm = llm.with_structured_output(SharedMoleculeAnalysis, include_raw=True)
        except Exception as exc:
            print(
                "[analyze_current] Structured output setup failed; using plain "
                f"text-only fallback: {type(exc).__name__}: {exc}"
            )
            structured_llm = None
        prompt_text = INITIAL_ANALYZE_USER_PROMPT.format(
            current_smiles=current_info.get("current_smiles"),
            iupac=current_info.get("current_iupac"),
            fragments=current_info.get("current_fragments"),
            optimization_goal=state.get("optimization_goal"),
            project_manager_brief=self._analysis_request_context(state, current_info),
        )
        if not self.config.enable_molecular_auxiliary_context:
            prompt_text = without_explicit_molecular_auxiliary_fields(prompt_text)
        fallback_messages = [
            SystemMessage(content=INITIAL_ANALYZE_SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ]
        if self.config.enable_molecular_auxiliary_context:
            visual_context = build_topology_visual_context(
                smiles=current_info.get("current_smiles"),
                visual_id="CURRENT_MOLECULE",
                label="current molecule for shared analysis",
                iupac=current_info.get("current_iupac"),
                fragments=current_info.get("current_fragments"),
                image_prefix="current_molecule_topology",
            )
            messages = [
                SystemMessage(content=INITIAL_ANALYZE_SYSTEM_PROMPT),
                HumanMessage(
                    content=[
                        text_block(prompt_text),
                        *visual_context.content_blocks,
                    ]
                ),
            ]
            primary_operation_name = "Analyze current molecule structured visual-context LLM call"
        else:
            messages = fallback_messages
            primary_operation_name = "Analyze current molecule structured SMILES-only LLM call"

        async def _invoke_structured_visual():
            return await structured_llm.ainvoke(messages)

        async def _invoke_structured_text():
            return await structured_llm.ainvoke(fallback_messages)

        async def _invoke_text_fallback():
            return await llm.ainvoke(fallback_messages)

        print("[analyze_current] Starting selected-candidate light analysis LLM call...")
        if structured_llm is None:
            response = await invoke_with_retry(
                _invoke_text_fallback,
                max_retries=3,
                base_delay=1.0,
                validate_fn=self._validate_text_response,
                operation_name="Analyze current molecule plain text-only fallback LLM call",
            )
            text = self._message_text(response)
            usage = getattr(response, "usage_metadata", None)
            print("[analyze_current] Selected-candidate light analysis completed.")
            return self._fallback_text_analysis(text), usage

        try:
            response = await invoke_with_retry(
                _invoke_structured_visual,
                max_retries=3,
                base_delay=1.0,
                validate_fn=self._validate_structured_analysis_response,
                operation_name=primary_operation_name,
            )
        except Exception as exc:
            print(
                "[analyze_current] Structured visual-context LLM call failed; retrying "
                "structured text-only fallback: "
                f"{type(exc).__name__}: {exc}"
            )
            try:
                response = await invoke_with_retry(
                    _invoke_structured_text,
                    max_retries=3,
                    base_delay=1.0,
                    validate_fn=self._validate_structured_analysis_response,
                    operation_name="Analyze current molecule structured text-only fallback LLM call",
                )
            except Exception as text_exc:
                print(
                    "[analyze_current] Structured text-only LLM call failed; retrying "
                    "plain text-only fallback: "
                    f"{type(text_exc).__name__}: {text_exc}"
                )
                response = await invoke_with_retry(
                    _invoke_text_fallback,
                    max_retries=3,
                    base_delay=1.0,
                    validate_fn=self._validate_text_response,
                    operation_name="Analyze current molecule plain text-only fallback LLM call",
                )
                text = self._message_text(response)
                usage = getattr(response, "usage_metadata", None)
                print("[analyze_current] Selected-candidate light analysis completed.")
                return self._fallback_text_analysis(text), usage

        parsed = self._parsed_structured_response(response)
        usage = self._structured_response_usage(response)
        print("[analyze_current] Selected-candidate light analysis completed.")
        return self._shared_analysis_to_dict(parsed), usage

    def _analysis_request_context(self, state: MCGSState, current_info: Dict[str, Any]) -> str:
        context_parts = []
        project_manager_brief = state.get("project_manager_brief")
        current_selection_context = current_info.get("current_selection_context")
        if project_manager_brief:
            context_parts.append(f"Project manager brief: {project_manager_brief}")
        if current_selection_context:
            context_parts.append(f"Current molecule selection context: {current_selection_context}")
        return "\n".join(str(part).strip() for part in context_parts if str(part).strip())

    def _empty_shared_analysis(self) -> Dict[str, Any]:
        return {
            "shared_analysis_summary": "No shared analysis available; proceed with minimal local diagnosis.",
            "shared_keep_fragments": [],
            "shared_modifiable_fragments": [],
            "shared_risk_alerts": [],
            "shared_priority_directions": [],
        }

    @staticmethod
    def _canonical_smiles_key(smiles: Any) -> str:
        ok, canonical, _error = validate_smiles(smiles)
        return canonical if ok and canonical else str(smiles or "").strip()

    @staticmethod
    def _validate_structured_analysis_response(result: Any) -> bool:
        parsed = AnalyzeCurrentNode._parsed_structured_response(result)
        if parsed is None:
            return False
        if isinstance(parsed, dict):
            summary = parsed.get("shared_analysis_summary")
        else:
            summary = getattr(parsed, "shared_analysis_summary", "")
        return bool(str(summary or "").strip())

    @staticmethod
    def _parsed_structured_response(result: Any) -> Any:
        if isinstance(result, dict):
            if "parsed" in result:
                return result.get("parsed")
            if "structured_response" in result:
                return result.get("structured_response")
        return result

    @staticmethod
    def _structured_response_usage(result: Any) -> Any:
        if isinstance(result, dict):
            raw = result.get("raw")
            usage = getattr(raw, "usage_metadata", None)
            if usage is not None:
                return usage
        return getattr(result, "usage_metadata", None)

    def _shared_analysis_to_dict(self, value: Any) -> Dict[str, Any]:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            data = model_dump()
        elif isinstance(value, dict):
            data = value
        else:
            data = {}

        return {
            "shared_analysis_summary": self._string_value(
                data.get("shared_analysis_summary")
            ) or "No shared analysis available.",
            "shared_keep_fragments": self._list_value(data.get("shared_keep_fragments")),
            "shared_modifiable_fragments": self._list_value(data.get("shared_modifiable_fragments")),
            "shared_risk_alerts": self._list_value(data.get("shared_risk_alerts")),
            "shared_priority_directions": self._list_value(data.get("shared_priority_directions")),
        }

    def _fallback_text_analysis(self, text: str) -> Dict[str, Any]:
        return {
            "shared_analysis_summary": text.strip() or "No shared analysis available.",
            "shared_keep_fragments": [],
            "shared_modifiable_fragments": [],
            "shared_risk_alerts": [],
            "shared_priority_directions": [],
        }

    @staticmethod
    def _string_value(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        return str(value or "").strip()

    @staticmethod
    def _list_value(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [
                line.strip().lstrip("-*0123456789. ").strip()
                for line in value.splitlines()
                if line.strip().lstrip("-*0123456789. ").strip()
            ]
        return []

    @staticmethod
    def _validate_text_response(result: Any) -> bool:
        return bool(AnalyzeCurrentNode._message_text(result).strip())

    @staticmethod
    def _message_text(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            if "content" in result:
                return AnalyzeCurrentNode._content_to_text(result.get("content"))
            return json.dumps(result, ensure_ascii=False)
        content = getattr(result, "content", None)
        if content is not None:
            return AnalyzeCurrentNode._content_to_text(content)
        return str(result)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part)
        return str(content or "")
