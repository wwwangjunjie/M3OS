"""Lightweight Auditor-derived role memory helpers for MCGS agents."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping


DEFAULT_MEMORY_LIMIT = 24
PROMPT_MEMORY_LIMIT = 8


def empty_agent_memory() -> Dict[str, List[Dict[str, Any]]]:
    """Return the default per-role memory container."""
    return {"rational": [], "creative": [], "critic": []}


def normalize_agent_memory(value: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Normalize an arbitrary state value into the expected memory mapping."""
    result = empty_agent_memory()
    if not isinstance(value, Mapping):
        return result
    for role in result:
        items = value.get(role)
        if isinstance(items, list):
            result[role] = [item for item in items if isinstance(item, dict)]
    return result


def append_agent_memory(
    memory: Any,
    role: str,
    items: Iterable[Mapping[str, Any]],
    *,
    limit: int = DEFAULT_MEMORY_LIMIT,
) -> Dict[str, List[Dict[str, Any]]]:
    """Append audited memory items for one role and keep only the newest items."""
    normalized = normalize_agent_memory(memory)
    role_key = _normalize_role(role)
    if role_key not in normalized:
        return normalized
    cleaned = [dict(item) for item in items if isinstance(item, Mapping)]
    if not cleaned:
        return normalized
    normalized[role_key] = (normalized[role_key] + cleaned)[-max(1, int(limit)):]
    return normalized


def format_agent_memory(
    memory: Any,
    role: str,
    *,
    limit: int = PROMPT_MEMORY_LIMIT,
    max_chars: int = 6000,
) -> str:
    """Format the newest role memory entries for dynamic agent user prompts."""
    normalized = normalize_agent_memory(memory)
    role_key = _normalize_role(role)
    items = normalized.get(role_key, [])
    if not items:
        return "No prior audited memory for this role."
    recent = items[-max(1, int(limit)):]
    text = json.dumps(recent, ensure_ascii=False, indent=2, default=str)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def generation_memory_items(
    molecules: Iterable[Mapping[str, Any]],
    *,
    round_index: Any,
    source_agent: str,
) -> List[Dict[str, Any]]:
    """Build memory items from audited generator output."""
    items: List[Dict[str, Any]] = []
    for molecule in molecules or []:
        if not isinstance(molecule, Mapping):
            continue
        items.append(
            {
                "memory_type": "audited_proposal",
                "round_index": round_index,
                "source_agent": source_agent,
                "smiles": molecule.get("SMILES") or molecule.get("smiles") or "",
                "modification_type": molecule.get("modification_type") or "",
                "rationale": molecule.get("rationale") or "",
                "confidence_score": molecule.get("confidence_score", 0.0),
            }
        )
    return items


def critic_memory_items(
    evaluations: Iterable[Mapping[str, Any]],
    *,
    round_index: Any,
) -> List[Dict[str, Any]]:
    """Build memory items from audited critic output or processed evaluations."""
    items: List[Dict[str, Any]] = []
    for evaluation in evaluations or []:
        if not isinstance(evaluation, Mapping):
            continue
        items.append(
            {
                "memory_type": "audited_evaluation",
                "round_index": round_index,
                "source_agent": "critic",
                "smiles": evaluation.get("smiles") or evaluation.get("SMILES") or "",
                "action": evaluation.get("action") or "",
                "score": evaluation.get("score", 0.0),
                "critic_score": evaluation.get("critic_score"),
                "rationale": (
                    evaluation.get("rationale")
                    or evaluation.get("critic_rationale")
                    or ""
                ),
                "properties": evaluation.get("properties") or {},
                "agent_type": evaluation.get("agent_type") or "",
            }
        )
    return items


def critic_feedback_items(
    evaluations: Iterable[Mapping[str, Any]],
    *,
    round_index: Any,
    agent_type: str,
) -> List[Dict[str, Any]]:
    """Build role-specific critic feedback for the generator that made a candidate."""
    role_agent = str(agent_type or "")
    items: List[Dict[str, Any]] = []
    for evaluation in evaluations or []:
        if not isinstance(evaluation, Mapping):
            continue
        if str(evaluation.get("agent_type") or "") != role_agent:
            continue
        items.append(
            {
                "memory_type": "critic_feedback",
                "round_index": round_index,
                "smiles": evaluation.get("smiles") or evaluation.get("SMILES") or "",
                "action": evaluation.get("action") or "",
                "score": evaluation.get("score", 0.0),
                "critic_score": evaluation.get("critic_score"),
                "critic_rationale": (
                    evaluation.get("critic_rationale")
                    or evaluation.get("rationale")
                    or ""
                ),
                "properties": evaluation.get("properties") or {},
            }
        )
    return items


def _normalize_role(role: str) -> str:
    text = str(role or "").strip().lower()
    aliases = {
        "rational_medicinal_designer": "rational",
        "rational_designer": "rational",
        "creative_molecule_explorer": "creative",
        "creative_explorer": "creative",
    }
    return aliases.get(text, text)
