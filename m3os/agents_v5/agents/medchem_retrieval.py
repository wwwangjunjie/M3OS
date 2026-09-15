"""Dedicated medicinal-chemistry retrieval agent."""

from __future__ import annotations

from typing import Any, List, Optional

from m3os.agents_v5.agents.base import BaseAgent
from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.prompts.system_loader import render_medchem_retrieval_system_prompt
from m3os.agents_v5.services.mcp_client import (
    MCPClientManager,
    MEDCHEM_RETRIEVAL_TOOL_NAMES,
    filter_tools_by_name,
)


class MedChemRetrievalAgent(BaseAgent):
    """Agent that owns raw medicinal-chemistry retrieval context and memory."""

    def __init__(
        self,
        config: AgentConfig,
        mcp_client: MCPClientManager,
        additional_context: Optional[str] = None,
    ):
        super().__init__(config, mcp_client, additional_context=additional_context)
        self._tools: Optional[List[Any]] = None

    @property
    def system_prompt(self) -> str:
        return render_medchem_retrieval_system_prompt(
            additional_context=self.additional_context,
        )

    @property
    def output_schema(self) -> type | None:
        return None

    @property
    def agent_role(self) -> str:
        return "medchem_retrieval"

    @property
    def inject_shared_medchem_memory(self) -> bool:
        return True

    async def get_tools(self) -> List[Any]:
        if self._tools is None:
            all_tools = self.mcp_client.get_all_agent_tools()
            self._tools = filter_tools_by_name(all_tools, MEDCHEM_RETRIEVAL_TOOL_NAMES)
        return self._tools

    async def answer(
        self,
        *,
        question: str,
        decision_context: str = "",
        answer_focus: str = "",
        requester_role: str = "",
        requester_context: Optional[dict[str, Any]] = None,
    ) -> str:
        if requester_context:
            self.set_event_context(
                agent_type="medchem_retrieval",
                requester_role=requester_role,
                round=requester_context.get("round") or requester_context.get("display_round"),
                display_round=requester_context.get("display_round"),
                round_index=requester_context.get("round_index"),
                mcgs_round=requester_context.get("mcgs_round"),
                display_mcgs_round=requester_context.get("display_mcgs_round"),
            )
        query = self._format_medchem_retrieval_prompt(
            question=question,
            decision_context=decision_context,
            answer_focus=answer_focus,
            requester_role=requester_role,
        )
        result = await self.invoke(query)
        text = self._medchem_answer_to_text(result)
        return self._compact_answer_text(text, 3000)


async def create_medchem_retrieval_agent(
    config: AgentConfig,
    mcp_client: MCPClientManager,
    additional_context: Optional[str] = None,
) -> MedChemRetrievalAgent:
    agent = MedChemRetrievalAgent(config, mcp_client, additional_context)
    await agent.build()
    return agent
