"""Prompt templates for M3OS.

System prompts live under `m3os.agents_v5.prompts.system` and are loaded by
`m3os.agents_v5.prompts.system_loader`. Runtime user/workflow prompts live
under `m3os.agents_v5.prompts.runtime`.
"""

from .runtime import (
    CREATIVE_EXPLORER_USER_PROMPT,
    RATIONAL_DESIGNER_USER_PROMPT,
    CRITIC_USER_PROMPT,
    EXTRACT_INFO_SYSTEM_PROMPT,
    INITIAL_ANALYZE_SYSTEM_PROMPT,
    INITIAL_ANALYZE_USER_PROMPT,
    SELECT_CANDIDATE_SYSTEM_PROMPT,
    SELECT_CANDIDATE_USER_PROMPT,
)

__all__ = [
    "CREATIVE_EXPLORER_USER_PROMPT",
    "RATIONAL_DESIGNER_USER_PROMPT",
    "CRITIC_USER_PROMPT",
    "EXTRACT_INFO_SYSTEM_PROMPT",
    "INITIAL_ANALYZE_SYSTEM_PROMPT",
    "INITIAL_ANALYZE_USER_PROMPT",
    "SELECT_CANDIDATE_SYSTEM_PROMPT",
    "SELECT_CANDIDATE_USER_PROMPT",
]
