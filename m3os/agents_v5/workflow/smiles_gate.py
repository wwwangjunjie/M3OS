"""Pre-audit SMILES validation and repair helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

# Legacy pre-finalization gate disabled; kept commented near its old helpers below.
# from m3os.agents_v5.core.trace_utils import extract_auditor_source_text
from m3os.agents_v5.services.smiles_validation import validate_smiles_batch_messages


SMILES_BATCH_TOOL = "validate_smiles_batch"

_FIELD_SMILES_RE = re.compile(
    r"""(?ix)
    \b(?:smiles|SMILES)\b
    \s*(?:[:=]|->)\s*
    [`"']?
    (?P<smiles>[^\s`"',;|<>]+)
    """
)
_JSON_SMILES_RE = re.compile(
    r"""(?ix)
    ["'](?:smiles|SMILES)["']\s*:\s*["']
    (?P<smiles>[^"']+)
    ["']
    """
)
_BACKTICK_TOKEN_RE = re.compile(r"`([^`\s]+)`")
_SMILES_TOKEN_CHARS_RE = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$:/\\%.*]+$")


# Legacy pre-finalization validation/repair path disabled.
# The active path validates the final auditor-facing summary after
# finalize_audit_summary(), via validate_finalized_auditor_source_smiles().
#
# async def validate_and_repair_smiles_before_audit(
#     agent: Any,
#     result: Any,
#     *,
#     audit_kind: str,
#     source_agent_type: str,
#     max_repair_attempts: int = 3,
# ) -> Dict[str, Any]:
#     """Validate final response SMILES and ask the source agent to repair invalid ones."""
#     result_dict = _result_as_dict(result)
#     gate: Dict[str, Any] = {
#         "tool": SMILES_BATCH_TOOL,
#         "audit_kind": audit_kind,
#         "source_agent_type": source_agent_type,
#         "checks": [],
#         "status": "not_checked",
#         "validated_smiles": [],
#         "invalid_smiles": {},
#         "repair_attempts": 0,
#         "extra_instruction": "",
#     }
#
#     for attempt in range(max_repair_attempts + 1):
#         smiles_list = extract_smiles_candidates_from_result(result_dict)
#         if not smiles_list:
#             gate["status"] = "no_smiles_found"
#             return _attach_gate(result_dict, gate)
#
#         validation = await _validate_smiles_batch(agent, smiles_list)
#         valid_smiles = [
#             smiles for smiles, message in validation.items()
#             if _is_valid_message(message)
#         ]
#         invalid = {
#             smiles: message for smiles, message in validation.items()
#             if not _is_valid_message(message)
#         }
#         gate["checks"].append(
#             {
#                 "attempt": attempt,
#                 "smiles": smiles_list,
#                 "validation": validation,
#                 "valid_count": len(valid_smiles),
#                 "invalid_count": len(invalid),
#             }
#         )
#         gate["validated_smiles"] = valid_smiles
#         gate["invalid_smiles"] = invalid
#
#         if not invalid:
#             gate["status"] = "all_valid"
#             gate["extra_instruction"] = _build_extra_instruction(gate)
#             return _attach_gate(result_dict, gate)
#
#         if attempt >= max_repair_attempts:
#             gate["status"] = "unrepaired_invalid_dropped"
#             gate["extra_instruction"] = _build_extra_instruction(gate)
#             return _attach_gate(result_dict, gate)
#
#         repair_invoker = getattr(agent, "invoke", None)
#         if not callable(repair_invoker):
#             gate["status"] = "invalid_unrepaired_no_agent_invoke"
#             gate["extra_instruction"] = _build_extra_instruction(gate)
#             return _attach_gate(result_dict, gate)
#
#         repair_prompt = _build_repair_prompt(
#             invalid,
#             valid_smiles,
#             audit_kind=audit_kind,
#             source_agent_type=source_agent_type,
#             attempt=attempt + 1,
#             max_attempts=max_repair_attempts,
#         )
#         result_dict = _result_as_dict(await repair_invoker(_append_user_prompt(result_dict, repair_prompt)))
#         gate["repair_attempts"] = attempt + 1
#
#     gate["status"] = "unrepaired_invalid_dropped"
#     gate["extra_instruction"] = _build_extra_instruction(gate)
#     return _attach_gate(result_dict, gate)


async def validate_finalized_auditor_source_smiles(
    agent: Any,
    finalized_source: Dict[str, Any],
    *,
    audit_kind: str,
    source_agent_type: str,
) -> Dict[str, Any]:
    """Validate SMILES in the final auditor-facing summary text."""
    source = dict(finalized_source or {})
    gate: Dict[str, Any] = {
        "tool": SMILES_BATCH_TOOL,
        "audit_kind": audit_kind,
        "source_agent_type": source_agent_type,
        "checks": [],
        "status": "not_checked",
        "validated_smiles": [],
        "invalid_smiles": {},
        "repair_attempts": 0,
        "extra_instruction": "",
        "validation_stage": "post_finalization",
    }

    smiles_list = extract_smiles_candidates_from_text(source.get("source_text", ""))
    if not smiles_list:
        gate["status"] = "no_smiles_found"
        source["smiles_validation_gate"] = gate
        return source

    validation = await _validate_smiles_batch(agent, smiles_list)
    valid_smiles = [
        smiles for smiles, message in validation.items()
        if _is_valid_message(message)
    ]
    invalid = {
        smiles: message for smiles, message in validation.items()
        if not _is_valid_message(message)
    }
    gate["checks"].append(
        {
            "attempt": 0,
            "smiles": smiles_list,
            "validation": validation,
            "valid_count": len(valid_smiles),
            "invalid_count": len(invalid),
        }
    )
    gate["validated_smiles"] = valid_smiles
    gate["invalid_smiles"] = invalid
    gate["status"] = "all_valid" if not invalid else "finalized_invalid_smiles"
    gate["extra_instruction"] = _build_extra_instruction(gate)
    source["smiles_validation_gate"] = gate
    return source


# Legacy pre-finalization source-result extraction disabled.
#
# def extract_smiles_candidates_from_result(result: Any) -> List[str]:
#     source_text, _warnings = extract_auditor_source_text(result)
#     if not source_text and isinstance(result, dict):
#         source_text = str(result.get("raw_response_text") or "")
#     return extract_smiles_candidates_from_text(source_text)


def extract_smiles_candidates_from_text(text: str) -> List[str]:
    """Extract likely molecule SMILES from audit-facing text."""
    table_candidates = _extract_table_smiles(text)
    if table_candidates:
        return _dedupe_smiles(_clean_smiles_candidate(item) for item in table_candidates)

    candidates: List[str] = []
    candidates.extend(match.group("smiles") for match in _JSON_SMILES_RE.finditer(text or ""))
    candidates.extend(match.group("smiles") for match in _FIELD_SMILES_RE.finditer(text or ""))
    candidates.extend(
        match.group(1) for match in _BACKTICK_TOKEN_RE.finditer(text or "")
        if _looks_like_smiles_token(match.group(1))
    )
    return _dedupe_smiles(_clean_smiles_candidate(item) for item in candidates)


def smiles_validation_extra_instruction(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    gate = result.get("smiles_validation_gate")
    if not isinstance(gate, dict):
        return ""
    return str(gate.get("extra_instruction") or "")


def _extract_table_smiles(text: str) -> List[str]:
    rows: List[List[str]] = []
    for line in str(text or "").splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 2:
            rows.append(cells)
    if not rows:
        return []

    candidates: List[str] = []
    active_smiles_index: int | None = None
    for cells in rows:
        lowered = [cell.strip().casefold() for cell in cells]
        if any("smiles" == cell or cell.endswith(" smiles") or "smiles" in cell for cell in lowered):
            active_smiles_index = next(
                (
                    index for index, cell in enumerate(lowered)
                    if "smiles" == cell or cell.endswith(" smiles") or "smiles" in cell
                ),
                None,
            )
            continue
        if all(set(cell.replace(" ", "")) <= {"-", ":"} for cell in cells):
            continue
        if active_smiles_index is not None and active_smiles_index < len(cells):
            candidates.append(cells[active_smiles_index])
    return candidates


async def _validate_smiles_batch(agent: Any, smiles_list: List[str]) -> Dict[str, str]:
    manager = getattr(agent, "mcp_client", None)
    invoker = getattr(manager, "invoke_tool", None)
    has_tool = getattr(manager, "has_tool", None)
    if callable(invoker) and (not callable(has_tool) or has_tool(SMILES_BATCH_TOOL)):
        try:
            result = await invoker(SMILES_BATCH_TOOL, {"smiles_list": smiles_list})
            normalized = _normalize_validation_result(result)
            if normalized:
                return normalized
        except Exception:
            return validate_smiles_batch_messages(smiles_list)
    return validate_smiles_batch_messages(smiles_list)


def _normalize_validation_result(result: Any) -> Dict[str, str]:
    if isinstance(result, dict):
        return {str(key): str(value) for key, value in result.items()}
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return {str(key): str(value) for key, value in parsed.items()}
    return {}


def _looks_like_smiles_token(token: Any) -> bool:
    text = str(token or "").strip()
    if not text or "_" in text or not _SMILES_TOKEN_CHARS_RE.match(text):
        return False
    return any(char.isdigit() or char in "[]()=#@/\\.+-" for char in text) or any(
        char in text for char in ("C", "N", "O", "S", "P", "F", "I", "B")
    )


def _is_valid_message(message: Any) -> bool:
    return str(message or "").strip() == "valid"


# Legacy source-agent repair prompt disabled.
#
# def _build_repair_prompt(
#     invalid: Dict[str, str],
#     valid_smiles: List[str],
#     *,
#     audit_kind: str,
#     source_agent_type: str,
#     attempt: int,
#     max_attempts: int,
# ) -> str:
#     role_note = (
#         "You are correcting copied candidate SMILES in your critic evaluation, not inventing new candidates. "
#         "Prefer exact valid candidate SMILES from the input list when the intended molecule is clear. "
#         if audit_kind == "critic_evaluations"
#         else (
#             "You are repairing generated molecule SMILES while preserving the same modification direction, "
#             "rationale, and confidence whenever chemically possible. "
#         )
#     )
#     return (
#         "[SMILES VALIDATION GATE]\n"
#         f"Source agent type: {source_agent_type}\n"
#         f"Repair attempt {attempt} of {max_attempts}.\n"
#         f"{role_note}\n"
#         "The following SMILES failed `validate_smiles_batch`; each value is the RDKit or pre-screen error:\n"
#         f"{json.dumps(invalid, ensure_ascii=False, indent=2)}\n\n"
#         f"Already valid SMILES from your current final list:\n{json.dumps(valid_smiles, ensure_ascii=False)}\n\n"
#         "Required actions:\n"
#         "1. Inspect each invalid SMILES and the modification/action/rationale it was meant to represent.\n"
#         "2. Replace it with a corrected chemically valid SMILES, or drop it if you cannot repair it confidently.\n"
#         "3. Call `validate_smiles_batch` with the complete revised final SMILES list before answering.\n"
#         "4. Final answer must list only validated SMILES. Do not repeat invalid SMILES anywhere in the final answer.\n"
#         "5. Keep the final answer concise and preserve the original fields needed by the downstream audit table.\n"
#     )


def _build_extra_instruction(gate: Dict[str, Any]) -> str:
    status = str(gate.get("status") or "")
    valid_smiles = [str(item) for item in gate.get("validated_smiles") or [] if str(item)]
    invalid = {
        str(key): str(value)
        for key, value in (gate.get("invalid_smiles") or {}).items()
        if str(key)
    }
    if status in {"not_checked", "no_smiles_found"}:
        return ""
    lines = [
        "SMILES validation gate: the final auditor-facing summary SMILES were checked with validate_smiles_batch.",
    ]
    if valid_smiles:
        lines.append(
            "Only include molecules whose SMILES are in this validated set if they are present in the preceding response: "
            f"{json.dumps(valid_smiles, ensure_ascii=False)}."
        )
    if invalid:
        lines.append(
            "Exclude these invalid or unrepaired SMILES entirely from the audit summary: "
            f"{json.dumps(invalid, ensure_ascii=False)}."
        )
    lines.append("Do not introduce any unvalidated replacement SMILES during audit finalization.")
    return "\n".join(lines).strip() + "\n"


# Legacy pre-finalization repair helpers disabled.
#
# def _append_user_prompt(result: Dict[str, Any], prompt: str) -> List[Any]:
#     messages = result.get("messages")
#     if isinstance(messages, list) and messages:
#         return [*messages, {"role": "user", "content": prompt}]
#
#     source_text = str(result.get("raw_response_text") or "").strip()
#     if source_text:
#         prompt = (
#             "[PREVIOUS RESPONSE]\n"
#             f"{source_text}\n\n"
#             f"{prompt}"
#         )
#     return [{"role": "user", "content": prompt}]
#
#
# def _attach_gate(result: Dict[str, Any], gate: Dict[str, Any]) -> Dict[str, Any]:
#     attached = dict(result)
#     attached["smiles_validation_gate"] = gate
#     return attached
#
#
# def _result_as_dict(result: Any) -> Dict[str, Any]:
#     if isinstance(result, dict):
#         return dict(result)
#     if isinstance(result, str):
#         return {"raw_response_text": result}
#     return {"raw_response_text": str(result or "")}


def _clean_smiles_candidate(value: Any) -> str:
    text = str(value or "").strip()
    text = text.strip("`\"'")
    while text and text[-1] in ".,;:":
        text = text[:-1].strip()
    return text


def _dedupe_smiles(values: Any) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped
