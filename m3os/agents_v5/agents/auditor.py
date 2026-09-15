"""Auditor agent for extracting workflow-safe structures from free-form outputs."""

from __future__ import annotations

import time
from typing import Any, Dict, Type

from pydantic import BaseModel

from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.core.models import CriticEvaluations, MoleculeOptimizations
from m3os.agents_v5.core.retry_utils import invoke_with_retry
from m3os.agents_v5.core.trace_utils import safe_serialize
from m3os.agents_v5.prompts.system_loader import render_auditor_system_prompt
from m3os.agents_v5.services.llm_factory import create_base_model


AUDITOR_SYSTEM_PROMPT = render_auditor_system_prompt()


class AuditorAgent:
    """Structured extractor used after free-form Rational, Creative, and Critic agents.

    The Auditor intentionally does not use DeepAgent, tools, skills, or role-scoped
    middleware. It is a narrow structured-output LLM call so the main workflow can
    keep using its existing Pydantic contracts.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._models: Dict[Type[BaseModel], Any] = {}

    async def audit_molecule_optimizations(
        self,
        *,
        source_text: str,
        source_agent_type: str,
    ) -> Dict[str, Any]:
        """Extract generated molecule candidates from a free-form generator response."""
        user_prompt = (
            "Extract optimized molecule proposals from the source text.\n\n"
            f"Source agent type: {source_agent_type}\n\n"
            "[SOURCE TEXT]\n"
            f"{source_text or ''}\n\n"
            "Return every explicitly proposed molecule as MoleculeOptimizations. "
            "For each item, extract smiles, modification_type, rationale, and confidence_score. "
            "If confidence is absent, use 0.0."
        )
        return await self._audit(
            schema=MoleculeOptimizations,
            user_prompt=user_prompt,
            source_text=source_text,
            audit_kind="molecule_optimizations",
        )

    async def audit_critic_evaluations(
        self,
        *,
        source_text: str,
        source_agent_type: str = "critic",
    ) -> Dict[str, Any]:
        """Extract critic evaluations from a free-form critic response."""
        user_prompt = (
            "Extract molecule evaluations from the source text.\n\n"
            f"Source agent type: {source_agent_type}\n\n"
            "[SOURCE TEXT]\n"
            f"{source_text or ''}\n\n"
            "Return every explicitly evaluated molecule as CriticEvaluations. "
            "For each item, extract smiles, action, rationale, score, properties, and agent_type. "
            "If score is absent, use 0.0. If properties are absent, use {}. "
            "For agent_type, copy only explicit creative/rational generator provenance; "
            "if absent, use an empty string and never use the source agent type. "
            "Do not include molecules whose SMILES strings are absent from the source text."
        )
        return await self._audit(
            schema=CriticEvaluations,
            user_prompt=user_prompt,
            source_text=source_text,
            audit_kind="critic_evaluations",
        )

    async def _audit(
        self,
        *,
        schema: Type[BaseModel],
        user_prompt: str,
        source_text: str,
        audit_kind: str,
    ) -> Dict[str, Any]:
        model = self._structured_model(schema)

        async def _invoke() -> Any:
            return await model.ainvoke(
                [
                    {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )

        start = time.perf_counter()
        raw_result = await invoke_with_retry(
            _invoke,
            max_retries=3,
            base_delay=1.0,
            validate_fn=self._validate_auditor_response,
            operation_name=f"Auditor {audit_kind}",
        )
        elapsed = round(time.perf_counter() - start, 3)
        usage = self._extract_usage_metadata(raw_result)
        structured = self._extract_structured(raw_result)
        warnings = self._build_warnings(structured, source_text, audit_kind)
        return {
            "structured_response": structured,
            "audit_warnings": warnings,
            "auditor_trace": {
                "audit_kind": audit_kind,
                "schema": schema.__name__,
                "elapsed_sec": elapsed,
                "usage": safe_serialize(usage),
                "structured_response_summary": self._summarize_structured(structured),
                "warnings": warnings,
            },
            "llm_usage": {
                "elapsed_sec": elapsed,
                "usage": usage,
            },
        }

    def _structured_model(self, schema: Type[BaseModel]) -> Any:
        cached = self._models.get(schema)
        if cached is not None:
            return cached
        base_model = create_base_model(self.config.llm)
        try:
            structured = base_model.with_structured_output(schema, include_raw=True)
        except TypeError:
            structured = base_model.with_structured_output(schema)
        self._models[schema] = structured
        return structured

    @staticmethod
    def _extract_structured(raw_result: Any) -> Any:
        if isinstance(raw_result, dict):
            parsed = raw_result.get("parsed")
            if parsed is not None:
                return parsed
            structured = raw_result.get("structured_response")
            if structured is not None:
                return structured
        return raw_result

    @staticmethod
    def _validate_auditor_response(raw_result: Any) -> bool:
        if raw_result is None:
            return False
        if isinstance(raw_result, dict):
            if "parsed" in raw_result:
                return raw_result.get("parsed") is not None
            if "structured_response" in raw_result:
                return raw_result.get("structured_response") is not None
            return bool(raw_result)
        return True

    @staticmethod
    def _extract_usage_metadata(raw_result: Any) -> Any:
        raw_message = raw_result.get("raw") if isinstance(raw_result, dict) else None
        usage = getattr(raw_message, "usage_metadata", None)
        if usage:
            return usage
        response_metadata = getattr(raw_message, "response_metadata", None)
        if isinstance(response_metadata, dict):
            return response_metadata.get("usage") or response_metadata.get("token_usage")
        return None

    @staticmethod
    def _summarize_structured(structured: Any) -> Dict[str, Any]:
        optimizations = getattr(structured, "optimizations", None)
        if isinstance(structured, dict):
            optimizations = structured.get("optimizations")
        if not isinstance(optimizations, list):
            return {"item_count": 0, "smiles": []}

        smiles_values = []
        for item in optimizations[:10]:
            smiles = item.get("smiles") if isinstance(item, dict) else getattr(item, "smiles", "")
            if smiles:
                smiles_values.append(str(smiles))
        return {
            "item_count": len(optimizations),
            "smiles": smiles_values,
        }

    @staticmethod
    def _build_warnings(structured: Any, source_text: str, audit_kind: str) -> list[str]:
        warnings: list[str] = []
        optimizations = getattr(structured, "optimizations", None)
        if isinstance(structured, dict):
            optimizations = structured.get("optimizations")
        if not isinstance(optimizations, list) or not optimizations:
            warnings.append(f"No {audit_kind} items were extracted from the source text.")
            return warnings

        source = source_text or ""
        for index, item in enumerate(optimizations, start=1):
            smiles = item.get("smiles") if isinstance(item, dict) else getattr(item, "smiles", "")
            if not smiles:
                warnings.append(f"Item {index} has no explicit SMILES.")
            elif str(smiles) not in source:
                warnings.append(f"Item {index} SMILES was not found verbatim in source text: {smiles}")

            if audit_kind == "molecule_optimizations":
                confidence = item.get("confidence_score") if isinstance(item, dict) else getattr(item, "confidence_score", None)
                if confidence in (None, 0, 0.0):
                    warnings.append(f"Item {index} is missing explicit confidence; defaulted to 0.0.")
            else:
                score = item.get("score") if isinstance(item, dict) else getattr(item, "score", None)
                if score in (None, 0, 0.0):
                    warnings.append(f"Item {index} is missing explicit score; defaulted to 0.0.")
        return warnings


async def create_auditor_agent(config: AgentConfig) -> AuditorAgent:
    """Factory function matching the other agent constructors."""
    return AuditorAgent(config)
