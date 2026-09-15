"""Shared cross-round screening context for generator agents."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping


def normalize_screening_context_history(value: Any) -> List[Dict[str, Any]]:
    """Return valid screening-context records in their original round order."""
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def format_screening_context_history(value: Any) -> str:
    """Format the complete accumulated screening history for an agent prompt."""
    history = normalize_screening_context_history(value)
    if not history:
        return "No prior Mol2Mol screening context."
    return json.dumps(history, ensure_ascii=False, indent=2, default=str)


def screening_context_updates(
    contexts: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize newly exported screening contexts for a graph-state update."""
    return [dict(item) for item in contexts if isinstance(item, Mapping)]
