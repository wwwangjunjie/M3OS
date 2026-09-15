"""Auditor empty-result recovery helpers for workflow nodes."""

from __future__ import annotations

from typing import Any, Dict

from m3os.agents_v5.core.trace_utils import (
    merge_auditor_source_warnings,
    safe_serialize,
    structured_field,
    structured_optimizations,
)
from m3os.agents_v5.services.smiles_validation import validate_smiles
from m3os.agents_v5.workflow.audit_finalization import (
    attach_finalization_metadata,
    finalized_auditor_source,
)
from m3os.agents_v5.workflow.smiles_gate import (
    smiles_validation_extra_instruction,
    validate_finalized_auditor_source_smiles,
)


DEFAULT_AUDITOR_EXTRACTION_ATTEMPTS = 3
DEFAULT_SUMMARY_REPAIR_ATTEMPTS = 3
DEFAULT_AGENT_RERUN_ATTEMPTS = 1
DEFAULT_FINALIZATION_SMILES_REPAIR_ATTEMPTS = 3


async def audit_with_empty_result_recovery(
    *,
    agent: Any,
    auditor_agent: Any,
    result: Any,
    audit_kind: str,
    source_agent_type: str,
    audit_method_name: str,
    rerun_query: str | None = None,
    extra_finalization_instruction: str = "",
    max_auditor_attempts: int = DEFAULT_AUDITOR_EXTRACTION_ATTEMPTS,
    max_summary_repair_attempts: int = DEFAULT_SUMMARY_REPAIR_ATTEMPTS,
    max_agent_reruns: int = DEFAULT_AGENT_RERUN_ATTEMPTS,
    max_finalization_smiles_repair_attempts: int = DEFAULT_FINALIZATION_SMILES_REPAIR_ATTEMPTS,
) -> Dict[str, Any]:
    """Audit a role-agent result and recover if the Auditor extracts no valid molecules.

    Recovery order:
    1. Ask the source agent's no-tool finalizer to build the audit summary.
    2. Validate the final auditor-facing SMILES and refinalize when needed.
    3. Give the Auditor several extraction attempts for the same finalized source.
    4. Ask the source agent's no-tool finalizer to rebuild the audit summary.
    5. If summaries are still unusable, rerun the source agent once when a query is
       available, then repeat the same local recovery loop.
    """
    recovery_trace: Dict[str, Any] = {
        "audit_kind": audit_kind,
        "source_agent_type": source_agent_type,
        "max_auditor_attempts": max_auditor_attempts,
        "max_summary_repair_attempts": max_summary_repair_attempts,
        "max_agent_reruns": max_agent_reruns,
        "max_finalization_smiles_repair_attempts": max_finalization_smiles_repair_attempts,
        "cycles": [],
        "status": "started",
    }
    working_result = result
    last_audit_result: Dict[str, Any] | None = None

    for agent_run_attempt in range(max_agent_reruns + 1):
        if agent_run_attempt > 0:
            rerun_result = await _rerun_source_agent(
                agent,
                rerun_query,
                audit_kind=audit_kind,
                source_agent_type=source_agent_type,
                attempt=agent_run_attempt,
                max_attempts=max_agent_reruns,
            )
            if rerun_result is None:
                recovery_trace["status"] = "agent_rerun_unavailable"
                break
            working_result = rerun_result

        base_extra_instruction = extra_finalization_instruction
        for summary_repair_attempt in range(max_summary_repair_attempts + 1):
            extra_instruction = base_extra_instruction
            if summary_repair_attempt > 0:
                extra_instruction = _join_instructions(
                    base_extra_instruction,
                    _summary_repair_instruction(
                        audit_kind=audit_kind,
                        source_agent_type=source_agent_type,
                        attempt=summary_repair_attempt,
                        max_attempts=max_summary_repair_attempts,
                    ),
                )

            finalized_source = await _finalized_source_with_smiles_validation(
                agent,
                working_result,
                audit_kind=audit_kind,
                source_agent_type=source_agent_type,
                extra_instruction=extra_instruction,
                max_attempts=max_finalization_smiles_repair_attempts,
            )
            audit_result, cycle_trace = await _audit_source_with_retries(
                auditor_agent,
                audit_method_name=audit_method_name,
                source_text=finalized_source.get("source_text", ""),
                source_agent_type=source_agent_type,
                max_attempts=max_auditor_attempts,
            )
            audit_result = merge_auditor_source_warnings(
                audit_result,
                finalized_source.get("source_warnings", []),
            )
            audit_result = attach_finalization_metadata(audit_result, finalized_source)
            _attach_source_agent_metadata(audit_result, working_result)
            audit_result["smiles_validation_gate"] = finalized_source.get("smiles_validation_gate")
            last_audit_result = audit_result

            cycle_trace.update(
                {
                    "agent_run_attempt": agent_run_attempt,
                    "summary_repair_attempt": summary_repair_attempt,
                    "source_chars": len(finalized_source.get("source_text", "") or ""),
                    "source_warnings": finalized_source.get("source_warnings", []),
                    "finalization_trace": safe_serialize(finalized_source.get("finalization_trace")),
                    "smiles_validation_gate": safe_serialize(finalized_source.get("smiles_validation_gate")),
                }
            )
            recovery_trace["cycles"].append(cycle_trace)

            if cycle_trace.get("valid_count", 0) > 0:
                recovery_trace["status"] = "success"
                recovery_trace["agent_reruns_used"] = agent_run_attempt
                recovery_trace["summary_repairs_used"] = summary_repair_attempt
                recovery_trace["auditor_attempts_used"] = cycle_trace.get("attempts_used", 0)
                _attach_recovery_trace(audit_result, recovery_trace)
                if agent_run_attempt or summary_repair_attempt or cycle_trace.get("attempts_used", 0) > 1:
                    _append_recovery_warning(
                        audit_result,
                        "Auditor recovery succeeded after retrying extraction and/or rebuilding the audit summary.",
                    )
                return audit_result

    recovery_trace["status"] = "exhausted"
    if last_audit_result is None:
        last_audit_result = {
            "structured_response": None,
            "audit_warnings": [],
            "auditor_trace": None,
            "llm_usage": None,
        }
        _attach_source_agent_metadata(last_audit_result, working_result)
    _attach_recovery_trace(last_audit_result, recovery_trace)
    _append_recovery_warning(
        last_audit_result,
        "Auditor recovery exhausted: no RDKit-valid audited molecules were extracted.",
    )
    return last_audit_result


async def _finalized_source_with_smiles_validation(
    agent: Any,
    result: Any,
    *,
    audit_kind: str,
    source_agent_type: str,
    extra_instruction: str,
    max_attempts: int,
) -> Dict[str, Any]:
    """Run audit finalization, then validate and repair that final text."""
    validation_instruction = ""
    finalized_source: Dict[str, Any] = {}

    for attempt in range(max_attempts + 1):
        finalized_source = await finalized_auditor_source(
            agent,
            result,
            audit_kind=audit_kind,
            source_agent_type=source_agent_type,
            extra_instruction=_join_instructions(extra_instruction, validation_instruction),
        )
        finalized_source = await validate_finalized_auditor_source_smiles(
            agent,
            finalized_source,
            audit_kind=audit_kind,
            source_agent_type=source_agent_type,
        )
        gate = finalized_source.get("smiles_validation_gate")
        if isinstance(gate, dict):
            gate["repair_attempts"] = attempt
        if not _needs_finalized_smiles_repair(gate) or attempt >= max_attempts:
            return finalized_source

        validation_instruction = _join_instructions(
            "The previous audit summary contained invalid SMILES after validation. "
            "Rewrite the same required audit table from the preceding work, preserving valid rows "
            "and removing or correcting invalid SMILES only when the correction is explicitly supported by the preceding work.",
            smiles_validation_extra_instruction(finalized_source),
        )

    return finalized_source


def _needs_finalized_smiles_repair(gate: Any) -> bool:
    if not isinstance(gate, dict):
        return False
    invalid = gate.get("invalid_smiles")
    return bool(invalid)


async def _audit_source_with_retries(
    auditor_agent: Any,
    *,
    audit_method_name: str,
    source_text: str,
    source_agent_type: str,
    max_attempts: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    audit_method = getattr(auditor_agent, audit_method_name)
    attempts: list[Dict[str, Any]] = []
    last_result: Dict[str, Any] | None = None
    last_validity: Dict[str, Any] = {"item_count": 0, "valid_count": 0, "invalid_items": []}

    for attempt in range(1, max_attempts + 1):
        audit_result = await audit_method(
            source_text=source_text,
            source_agent_type=source_agent_type,
        )
        validity = _structured_validity(audit_result.get("structured_response"))
        attempts.append(
            {
                "attempt": attempt,
                "item_count": validity["item_count"],
                "valid_count": validity["valid_count"],
                "invalid_items": validity["invalid_items"],
                "auditor_trace": safe_serialize(audit_result.get("auditor_trace")),
                "audit_warnings": safe_serialize(audit_result.get("audit_warnings", [])),
            }
        )
        last_result = audit_result
        last_validity = validity
        if validity["valid_count"] > 0:
            break

    if last_result is None:
        last_result = {
            "structured_response": None,
            "audit_warnings": [],
            "auditor_trace": None,
            "llm_usage": None,
        }

    return last_result, {
        "attempts": attempts,
        "attempts_used": len(attempts),
        "item_count": last_validity["item_count"],
        "valid_count": last_validity["valid_count"],
        "invalid_items": last_validity["invalid_items"],
    }


def _structured_validity(structured: Any) -> Dict[str, Any]:
    optimizations = structured_optimizations(structured)
    invalid_items: list[Dict[str, Any]] = []
    valid_count = 0
    seen_canonical: set[str] = set()

    for index, item in enumerate(optimizations, start=1):
        smiles = structured_field(item, "smiles", "")
        ok, canonical, error = validate_smiles(smiles)
        if not ok:
            invalid_items.append(
                {
                    "index": index,
                    "smiles": str(smiles or ""),
                    "error": error or "invalid SMILES",
                }
            )
            continue
        dedupe_key = canonical or str(smiles)
        if dedupe_key in seen_canonical:
            invalid_items.append(
                {
                    "index": index,
                    "smiles": str(smiles or ""),
                    "canonical_smiles": canonical,
                    "error": "duplicate canonical SMILES",
                }
            )
            continue
        seen_canonical.add(dedupe_key)
        valid_count += 1

    return {
        "item_count": len(optimizations),
        "valid_count": valid_count,
        "invalid_items": invalid_items[:10],
    }


async def _rerun_source_agent(
    agent: Any,
    rerun_query: str | None,
    *,
    audit_kind: str,
    source_agent_type: str,
    attempt: int,
    max_attempts: int,
) -> Any | None:
    invoker = getattr(agent, "invoke", None)
    if not callable(invoker) or not rerun_query:
        return None

    prompt = (
        f"{rerun_query}\n\n"
        "[AUDIT RECOVERY RERUN]\n"
        f"Source agent type: {source_agent_type}\n"
        f"Full rerun attempt {attempt} of {max_attempts}.\n"
        "The Auditor could not extract any RDKit-valid molecules after repeated extraction "
        "and audit-summary repair attempts. Redo the current task from your original role. "
        "Keep the medicinal-chemistry analysis grounded in the task context, then end with "
        "a concise final answer containing explicit valid SMILES and the fields required by "
        f"the downstream {audit_kind} audit summary. Do not return placeholders or malformed SMILES."
    )
    return await invoker(prompt)


def _summary_repair_instruction(
    *,
    audit_kind: str,
    source_agent_type: str,
    attempt: int,
    max_attempts: int,
) -> str:
    if audit_kind == "critic_evaluations":
        schema_note = (
            "Rebuild the table with exactly: smiles, action, rationale, score, properties, agent_type. "
            "Include one row per evaluated candidate and copy the generator provenance only when explicit."
        )
    else:
        schema_note = (
            "Rebuild the table with exactly: smiles, modification_type, rationale, confidence_score. "
            "Include one row per final proposed molecule, capped at 7 rows."
        )
    return (
        "[AUDITOR SUMMARY REPAIR]\n"
        f"Source agent type: {source_agent_type}\n"
        f"Summary repair attempt {attempt} of {max_attempts}.\n"
        "The Auditor could not extract any RDKit-valid molecules from the previous audit summary. "
        "The likely cause is a missing, malformed, or non-compliant summary table rather than the "
        "underlying reasoning.\n"
        f"{schema_note}\n"
        "Use only molecules and facts already present in your preceding work. "
        "Every SMILES cell must contain a real molecule SMILES, not prose, a property name, a score, "
        "a markdown separator, or a placeholder. Do not add prose outside the table."
    )


def _join_instructions(*parts: str) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip()).strip()


def _attach_source_agent_metadata(audit_result: Dict[str, Any], source_result: Any) -> None:
    source_dict = source_result if isinstance(source_result, dict) else {}
    audit_result["source_agent_result"] = source_result
    audit_result["raw_agent_response"] = source_dict.get("raw_response_text")
    audit_result["source_agent_llm_usage"] = source_dict.get("llm_usage")
    audit_result["source_agent_invocation_trace"] = source_dict.get("invocation_trace")
    audit_result["smiles_validation_gate"] = source_dict.get("smiles_validation_gate")


def _attach_recovery_trace(audit_result: Dict[str, Any], recovery_trace: Dict[str, Any]) -> None:
    audit_result["audit_recovery_trace"] = recovery_trace


def _append_recovery_warning(audit_result: Dict[str, Any], warning: str) -> None:
    warnings = audit_result.get("audit_warnings")
    audit_warnings = list(warnings) if isinstance(warnings, list) else []
    if warning not in audit_warnings:
        audit_warnings.append(warning)
    audit_result["audit_warnings"] = audit_warnings

    trace = audit_result.get("auditor_trace")
    if isinstance(trace, dict):
        trace_warnings = trace.get("warnings")
        merged = list(trace_warnings) if isinstance(trace_warnings, list) else []
        if warning not in merged:
            merged.append(warning)
        trace["warnings"] = merged
