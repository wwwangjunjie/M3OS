"""Utilities for capturing and rendering sub-agent traces."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


AUDITOR_CONTEXT_MARKERS = (
    "[PROMPT CONTEXT]",
    "## [Current Task Data]",
    "[Current Task Data]",
    "[SHARED CURRENT ANALYSIS CONTEXT]",
)


def safe_serialize(value: Any) -> Any:
    """Convert arbitrary values into JSON-friendly data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        if _is_image_content_block(value):
            redacted = {
                str(key): safe_serialize(val)
                for key, val in value.items()
                if key != "base64"
            }
            if "base64" in value:
                redacted["base64"] = f"[redacted image base64: {len(str(value.get('base64') or ''))} chars]"
            return redacted
        return {str(key): safe_serialize(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [safe_serialize(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return safe_serialize(model_dump(mode="json"))
        except TypeError:
            try:
                return safe_serialize(model_dump())
            except (AttributeError, TypeError):
                pass
        except AttributeError:
            pass

    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            return safe_serialize(dict_method())
        except TypeError:
            pass

    return repr(value)


def _is_image_content_block(value: Dict[Any, Any]) -> bool:
    block_type = str(value.get("type") or "").lower()
    return block_type == "image" and "base64" in value


def structured_field(value: Any, field: str, default: Any = None) -> Any:
    """Read a field from either a Pydantic-like object or a plain dict."""
    if isinstance(value, dict):
        return value.get(field, default)
    return getattr(value, field, default)


def structured_optimizations(structured: Any) -> list[Any]:
    """Return the optimizations list from structured output in dict/model form."""
    optimizations = structured_field(structured, "optimizations", [])
    return optimizations if isinstance(optimizations, list) else []


def extract_auditor_source_text(value: Any) -> tuple[str, list[str]]:
    """Return only the final free-form agent response for auditor extraction."""
    warnings: list[str] = []

    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        raw_text = value.get("raw_response_text")
        if isinstance(raw_text, str) and raw_text.strip():
            text = raw_text
        else:
            text = _extract_last_assistant_text(value.get("messages"))
            if not text:
                warnings.append("No final assistant text found for auditor input.")
    else:
        text = ""
        warnings.append("Unsupported agent result type for auditor input; no fallback serialization used.")

    cleaned, clean_warnings = strip_auditor_prompt_context(text)
    warnings.extend(clean_warnings)
    return cleaned, warnings


def strip_auditor_prompt_context(text: str) -> tuple[str, list[str]]:
    """Remove accidentally included prompt context from auditor source text."""
    if not text:
        return "", []

    warnings: list[str] = []
    cleaned = text.strip()
    if any(marker in cleaned for marker in AUDITOR_CONTEXT_MARKERS):
        warnings.append("Auditor source text contained prompt-context markers; stripped context before audit.")

    source_marker = "[SOURCE TEXT]"
    if source_marker in cleaned:
        cleaned = cleaned.rsplit(source_marker, 1)[1].strip()
        for trailing in (
            "\n\nReturn every explicitly proposed molecule",
            "\n\nReturn every explicitly evaluated molecule",
            "\n\nFor each item, extract",
        ):
            index = cleaned.find(trailing)
            if index >= 0:
                cleaned = cleaned[:index].strip()
                break
        return cleaned, warnings

    marker_positions = [
        cleaned.find(marker)
        for marker in AUDITOR_CONTEXT_MARKERS
        if marker in cleaned
    ]
    if marker_positions:
        cleaned = cleaned[: min(marker_positions)].strip()

    return cleaned, warnings


def _extract_last_assistant_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        if _is_tool_message(message):
            continue
        role = _message_role(message)
        if role and role not in {"assistant", "ai"}:
            continue
        text = _message_content_to_text(getattr(message, "content", None))
        if text:
            return text
    return ""


def _is_tool_message(message: Any) -> bool:
    message_type = str(getattr(message, "type", "") or "").lower()
    role = str(getattr(message, "role", "") or "").lower()
    return message_type == "tool" or role == "tool" or getattr(message, "tool_call_id", None) is not None


def _message_role(message: Any) -> str:
    role = getattr(message, "role", None)
    if isinstance(role, str):
        return role.lower()
    message_type = getattr(message, "type", None)
    if isinstance(message_type, str):
        return message_type.lower()
    return ""


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return ""


def merge_auditor_source_warnings(audit_result: Dict[str, Any], warnings: list[str]) -> Dict[str, Any]:
    """Attach source-cleanup warnings to the auditor result without expanding trace payloads."""
    if not warnings:
        return audit_result

    existing = audit_result.get("audit_warnings")
    audit_warnings = list(existing) if isinstance(existing, list) else []
    audit_warnings.extend(warnings)
    audit_result["audit_warnings"] = audit_warnings

    trace = audit_result.get("auditor_trace")
    if isinstance(trace, dict):
        trace_warnings = trace.get("warnings")
        merged_trace_warnings = list(trace_warnings) if isinstance(trace_warnings, list) else []
        merged_trace_warnings.extend(warnings)
        trace["warnings"] = merged_trace_warnings
    return audit_result


def serialize_message(message: Any, index: Optional[int] = None) -> Dict[str, Any]:
    """Serialize a LangChain/deepagents message object."""
    payload = {
        "index": index,
        "message_type": getattr(message, "type", message.__class__.__name__),
        "role": getattr(message, "role", None),
        "id": getattr(message, "id", None),
        "name": getattr(message, "name", None),
        "content": safe_serialize(getattr(message, "content", None)),
        "tool_calls": safe_serialize(getattr(message, "tool_calls", None)),
        "invalid_tool_calls": safe_serialize(getattr(message, "invalid_tool_calls", None)),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "status": getattr(message, "status", None),
        "artifact": safe_serialize(getattr(message, "artifact", None)),
        "additional_kwargs": safe_serialize(getattr(message, "additional_kwargs", None)),
        "response_metadata": safe_serialize(getattr(message, "response_metadata", None)),
        "usage_metadata": safe_serialize(getattr(message, "usage_metadata", None)),
    }
    return {key: value for key, value in payload.items() if value is not None}


def build_invocation_trace(
    agent_name: str,
    input_messages: Any,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a JSON-friendly trace for one agent invocation."""
    messages = result.get("messages") if isinstance(result, dict) else None
    serialized_messages = [
        serialize_message(message, index=index)
        for index, message in enumerate(messages or [])
    ]

    raw_result = safe_serialize(result if isinstance(result, dict) else {"result": result})

    return {
        "agent_name": agent_name,
        "input_messages": safe_serialize(input_messages),
        "messages": serialized_messages,
        "structured_response": safe_serialize(result.get("structured_response") if isinstance(result, dict) else None),
        "llm_usage": safe_serialize(result.get("llm_usage") if isinstance(result, dict) else None),
        "raw_result": raw_result,
    }


def format_trace_block(label: str, trace: Any) -> str:
    """Render a stored trace as a readable JSON block."""
    body = json.dumps(safe_serialize(trace), ensure_ascii=False, indent=2)
    return (
        f"\n=== {label} Full Trace ===\n"
        f"{body}\n"
        f"===[END] {label} Full Trace ==="
    )
