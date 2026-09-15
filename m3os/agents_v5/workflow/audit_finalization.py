"""Audit-source finalization helpers for workflow nodes."""

from __future__ import annotations

from typing import Any, Dict

from m3os.agents_v5.core.trace_utils import extract_auditor_source_text


async def finalized_auditor_source(
    agent: Any,
    result: Any,
    *,
    audit_kind: str,
    source_agent_type: str,
    extra_instruction: str = "",
) -> Dict[str, Any]:
    """Return the exact text that should be given to the Auditor.

    Real role agents implement ``finalize_audit_summary`` and produce one last
    no-tool Audit-friendly summary. The fallback keeps tests or legacy agent
    doubles working, while explicitly marking that finalization was unavailable.
    """
    finalizer = getattr(agent, "finalize_audit_summary", None)
    if callable(finalizer):
        return await finalizer(
            result,
            audit_kind=audit_kind,
            source_agent_type=source_agent_type,
            extra_instruction=extra_instruction,
        )

    source_text, warnings = extract_auditor_source_text(result)
    warnings.append(
        "Audit finalization was unavailable on this agent; used the original final response."
    )
    return {
        "source_text": source_text,
        "source_warnings": warnings,
        "finalization_trace": None,
        "llm_usage": None,
    }


def attach_finalization_metadata(
    audit_result: Dict[str, Any],
    finalized_source: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach auditor-source metadata without changing the structured payload."""
    audit_result["audit_source_text"] = finalized_source.get("source_text", "")
    audit_result["audit_finalization_trace"] = finalized_source.get("finalization_trace")
    audit_result["finalization_llm_usage"] = finalized_source.get("llm_usage")
    return audit_result
