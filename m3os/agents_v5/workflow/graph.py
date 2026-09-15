"""Expansion workflow graph definition for M3OS MCGS."""

from langgraph.graph import StateGraph, START, END

from m3os.agents_v5.core.state import MCGSState
from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.agents.critic import CriticAgent
from m3os.agents_v5.agents.auditor import AuditorAgent
from m3os.agents_v5.agents.creative_explorer import CreativeMoleculeExplorerAgent
from m3os.agents_v5.agents.rational_designer import (
    RationalMedicinalDesignerAgent,
)
from m3os.agents_v5.workflow.nodes import (
    CreativeExplorerNode,
    RationalDesignerNode,
    MoleculesAggregatorNode,
    EvaluateAndUpdateNode,
)


class MCGSExpansionWorkflowBuilder:
    """Builder for one MCGS expansion step.

    The expansion workflow is intentionally narrow: generator agents propose
    molecules, their outputs are aggregated, and the critic evaluates them while
    updating the graph. Task preparation, candidate selection, shared analysis,
    finish checks, and user interaction are owned by the main agent/session
    layer.
    """

    def __init__(
        self,
        config: AgentConfig,
        creative_agent: CreativeMoleculeExplorerAgent,
        rational_agent: RationalMedicinalDesignerAgent,
        critic_agent: CriticAgent,
        auditor_agent: AuditorAgent | None = None,
        enabled_generators: list[str] | tuple[str, ...] | None = None,
    ):
        self.config = config
        self.creative_agent = creative_agent
        self.rational_agent = rational_agent
        self.critic_agent = critic_agent
        self.auditor_agent = auditor_agent or AuditorAgent(config)
        self.enabled_generators = self._normalize_generators(enabled_generators)

    @staticmethod
    def _normalize_generators(enabled_generators: list[str] | tuple[str, ...] | None) -> set[str]:
        if not enabled_generators:
            return {"creative", "rational"}
        normalized = {str(item).strip().lower() for item in enabled_generators if str(item).strip()}
        aliases = {
            "creative_molecule_explorer": "creative",
            "creative_explorer": "creative",
            "rational_medicinal_designer": "rational",
            "rational_designer": "rational",
        }
        normalized = {aliases.get(item, item) for item in normalized}
        allowed = normalized & {"creative", "rational"}
        return allowed or {"creative", "rational"}

    def build(self) -> StateGraph:
        workflow = StateGraph(MCGSState)

        creative_explorer_node = CreativeExplorerNode(self.config, self.creative_agent, self.auditor_agent)
        rational_designer_node = RationalDesignerNode(self.config, self.rational_agent, self.auditor_agent)
        aggregator_node = MoleculesAggregatorNode()
        evaluate_update_node = EvaluateAndUpdateNode(self.config, self.critic_agent, self.auditor_agent)

        if "creative" in self.enabled_generators:
            workflow.add_node("creative_explorer", creative_explorer_node)
        if "rational" in self.enabled_generators:
            workflow.add_node("rational_designer", rational_designer_node)
        workflow.add_node("molecules_aggregator", aggregator_node)
        workflow.add_node("evaluate_and_update", evaluate_update_node)

        if "creative" in self.enabled_generators:
            workflow.add_edge(START, "creative_explorer")
            workflow.add_edge("creative_explorer", "molecules_aggregator")
        if "rational" in self.enabled_generators:
            workflow.add_edge(START, "rational_designer")
            workflow.add_edge("rational_designer", "molecules_aggregator")
        workflow.add_edge("molecules_aggregator", "evaluate_and_update")
        workflow.add_edge("evaluate_and_update", END)

        return workflow

    def compile(self):
        workflow = self.build()
        return workflow.compile()

def create_mcgs_expansion_workflow(
    config: AgentConfig,
    creative_agent: CreativeMoleculeExplorerAgent,
    rational_agent: RationalMedicinalDesignerAgent,
    critic_agent: CriticAgent,
    auditor_agent: AuditorAgent | None = None,
    enabled_generators: list[str] | tuple[str, ...] | None = None,
):
    """Create and compile the core MCGS expansion workflow.

    The caller must provide a fully prepared ``MCGSState`` containing the
    current molecule, graph, optimization context, retrieved knowledge, and
    shared analysis. This workflow only performs generation, aggregation, and
    critic evaluation/update.
    """
    builder = MCGSExpansionWorkflowBuilder(
        config=config,
        creative_agent=creative_agent,
        rational_agent=rational_agent,
        critic_agent=critic_agent,
        auditor_agent=auditor_agent,
        enabled_generators=enabled_generators,
    )
    return builder.compile()
