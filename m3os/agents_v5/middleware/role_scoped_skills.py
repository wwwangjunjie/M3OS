"""Role-scoped skills middleware for M3OS agents.

This wraps Deep Agents' SkillsMiddleware so we can keep a single flat
`skills/` source directory while exposing only the role-appropriate skills to
an agent's Skills System prompt.
"""

from __future__ import annotations

from typing import Any, Callable

from deepagents.middleware.skills import SkillsMiddleware, SkillsStateUpdate
from langchain.agents.middleware.types import ContextT, ModelRequest, ModelResponse, ResponseT


class RoleScopedSkillsMiddleware(SkillsMiddleware):
    """Skills middleware that filters loaded skills by allowed skill names."""

    def __init__(self, *, backend: Any, sources: list[str], allowed_skill_names: list[str]) -> None:
        super().__init__(backend=backend, sources=sources)
        self.allowed_skill_names = set(allowed_skill_names)

    def _filter_update(self, update: SkillsStateUpdate | None) -> SkillsStateUpdate | None:
        if update is None:
            return None
        filtered = [
            skill for skill in update["skills_metadata"] if skill["name"] in self.allowed_skill_names
        ]
        return SkillsStateUpdate(skills_metadata=filtered)

    def before_agent(self, state, runtime, config):  # type: ignore[override]
        update = super().before_agent(state, runtime, config)
        return self._filter_update(update)

    async def abefore_agent(self, state, runtime, config):  # type: ignore[override]
        update = await super().abefore_agent(state, runtime, config)
        return self._filter_update(update)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        request = request.override(
            state={
                **request.state,
                "skills_metadata": [
                    skill
                    for skill in request.state.get("skills_metadata", [])
                    if skill["name"] in self.allowed_skill_names
                ],
            }
        )
        return super().wrap_model_call(request, handler)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Any],
    ) -> ModelResponse[ResponseT]:
        request = request.override(
            state={
                **request.state,
                "skills_metadata": [
                    skill
                    for skill in request.state.get("skills_metadata", [])
                    if skill["name"] in self.allowed_skill_names
                ],
            }
        )
        return await super().awrap_model_call(request, handler)
