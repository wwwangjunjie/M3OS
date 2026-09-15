"""M3OS molecular optimization multi-agent system.

This package provides a refactored, production-ready implementation of the
Monte Carlo Graph Search (MCGS) molecular optimization system using LangGraph
and LangChain 1.0.

Key Components:
    - core: State definitions, Pydantic models, and configuration
    - agents: Creative, Rational, and Critic agent implementations
    - workflow: LangGraph expansion workflow nodes and graph definition
    - services: LLM factory, MCP client, knowledge context, utilities
    - tools: MCGS tool wrappers
    - prompts: Prompt templates

"""

from m3os.agents_v5.core import (
    # State
    MCGSState,
    # Generation Outputs
    MoleculeOptimization,
    MoleculeOptimizations,
    # Critic Outputs
    CriticMoleculeEvaluation,
    CriticEvaluations,
    # Node Outputs
    ExtractInitialInfo,
    SelectPromisingMolecule,
    ReportModificationSite,
    ReportStrategyGroup,
    ReportDesignedMolecule,
    ReportFirstRoundCandidate,
    OptimizationReport,
    # Config
    AgentConfig,
    LLMConfig,
    MCPConfig,
    PathsConfig,
    DeepAgentConfig,
)

from m3os.agents_v5.agents import (
    BaseAgent,
    CreativeMoleculeExplorerAgent,
    create_creative_molecule_explorer_agent,
    RationalMedicinalDesignerAgent,
    create_rational_medicinal_designer_agent,
    CriticAgent,
    create_critic_agent,
    AuditorAgent,
    create_auditor_agent,
    MedChemRetrievalAgent,
    create_medchem_retrieval_agent,
    M3OSChatSession,
    create_main_agent_session,
)

from m3os.agents_v5.workflow import (
    MCGSExpansionWorkflowBuilder,
    create_mcgs_expansion_workflow,
    InitGraphNode,
    SelectBestCandidateNode,
    AnalyzeCurrentNode,
    CreativeExplorerNode,
    RationalDesignerNode,
    MoleculesAggregatorNode,
    EvaluateAndUpdateNode,
)

from m3os.agents_v5.services import (
    LLMFactory,
    LLMProvider,
    ClaudeProvider,
    GeminiProvider,
    KimiProvider,
    create_base_model,
    get_middleware,
    MCPClientManager,
    filter_tools_by_name,
    TOOL_NAMES_CREATIVE,
    TOOL_NAMES_RATIONAL,
    TOOL_NAMES_CRITIC,
    TOOL_NAMES_MCP,
    PROTEIN_CONTEXT_TOOL_NAMES,
    NESSO_ACTIVITY_TOOL_NAMES,
    WIKI_TOOL_NAMES,
    WIKI_USABLE_TOOL_NAMES,
    WIKI_RAW_SOURCE_TOOL_NAMES,
    FAST_MEDCHEM_TOOL_NAMES,
    MEDCHEM_RETRIEVAL_TOOL_NAMES,
    RESEARCH_LITERATURE_TOOL_NAMES,
    MAIN_CONTEXT_LOOKUP_TOOL_NAMES,
    DocumentRAGService,
    DocumentIngestionResult,
    KnowledgeContextService,
    M3OSWikiKnowledgeService,
    MoleculeUtils,
    TaskPreparationService,
)

from m3os.agents_v5.tools import (
    mcgs_add_root_node,
    mcgs_get_top_5_nodes,
    mcgs_add_optimized_molecules,
    mcgs_backpropagate,
    mcgs_visualize_graph,
    mcgs_is_next_round_required,
    mcgs_export_graph_to_csv,
)

__version__ = "3.0.0"
__all__ = [
    # Version
    "__version__",
    # State
    "MCGSState",
    # Generation Outputs
    "MoleculeOptimization",
    "MoleculeOptimizations",
    # Critic Outputs
    "CriticMoleculeEvaluation",
    "CriticEvaluations",
    # Node Outputs
    "ExtractInitialInfo",
    "SelectPromisingMolecule",
    "ReportModificationSite",
    "ReportStrategyGroup",
    "ReportDesignedMolecule",
    "ReportFirstRoundCandidate",
    "OptimizationReport",
    # Config
    "AgentConfig",
    "LLMConfig",
    "MCPConfig",
    "PathsConfig",
    "DeepAgentConfig",
    # Agents
    "BaseAgent",
    "CreativeMoleculeExplorerAgent",
    "create_creative_molecule_explorer_agent",
    "RationalMedicinalDesignerAgent",
    "create_rational_medicinal_designer_agent",
    "CriticAgent",
    "create_critic_agent",
    "AuditorAgent",
    "create_auditor_agent",
    "MedChemRetrievalAgent",
    "create_medchem_retrieval_agent",
    "M3OSChatSession",
    "create_main_agent_session",
    # Workflow
    "MCGSExpansionWorkflowBuilder",
    "create_mcgs_expansion_workflow",
    "InitGraphNode",
    "SelectBestCandidateNode",
    "AnalyzeCurrentNode",
    "CreativeExplorerNode",
    "RationalDesignerNode",
    "MoleculesAggregatorNode",
    "EvaluateAndUpdateNode",
    # Services
    "LLMFactory",
    "LLMProvider",
    "ClaudeProvider",
    "GeminiProvider",
    "KimiProvider",
    "create_base_model",
    "get_middleware",
    "MCPClientManager",
    "filter_tools_by_name",
    "TOOL_NAMES_CREATIVE",
    "TOOL_NAMES_RATIONAL",
    "TOOL_NAMES_CRITIC",
    "TOOL_NAMES_MCP",
    "PROTEIN_CONTEXT_TOOL_NAMES",
    "NESSO_ACTIVITY_TOOL_NAMES",
    "WIKI_TOOL_NAMES",
    "WIKI_USABLE_TOOL_NAMES",
    "WIKI_RAW_SOURCE_TOOL_NAMES",
    "FAST_MEDCHEM_TOOL_NAMES",
    "MEDCHEM_RETRIEVAL_TOOL_NAMES",
    "RESEARCH_LITERATURE_TOOL_NAMES",
    "MAIN_CONTEXT_LOOKUP_TOOL_NAMES",
    "DocumentRAGService",
    "DocumentIngestionResult",
    "KnowledgeContextService",
    "M3OSWikiKnowledgeService",
    "MoleculeUtils",
    "TaskPreparationService",
    # Tools
    "mcgs_add_root_node",
    "mcgs_get_top_5_nodes",
    "mcgs_add_optimized_molecules",
    "mcgs_backpropagate",
    "mcgs_visualize_graph",
    "mcgs_is_next_round_required",
    "mcgs_export_graph_to_csv",
]
