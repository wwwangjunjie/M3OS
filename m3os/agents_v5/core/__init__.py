"""Core exports for M3OS.

This module provides the foundational types, models, and configuration
for the molecular optimization multi-agent system.
"""

from m3os.agents_v5.core.state import MCGSState
from m3os.agents_v5.core.models import (
    MoleculeOptimization,
    MoleculeOptimizations,
    CriticMoleculeEvaluation,
    CriticEvaluations,
    ExtractInitialInfo,
    SelectPromisingMolecule,
    SharedMoleculeAnalysis,
    ReportModificationSite,
    ReportStrategyGroup,
    ReportDesignedMolecule,
    ReportFirstRoundCandidate,
    OptimizationReport,
)
from m3os.agents_v5.core.config import (
    AgentConfig,
    LLMConfig,
    MCPConfig,
    PathsConfig,
    DeepAgentConfig,
)
from m3os.agents_v5.core.trace_utils import (
    build_invocation_trace,
    format_trace_block,
    safe_serialize,
    serialize_message,
)

__all__ = [
    # State
    "MCGSState",
    # Generation Outputs
    "MoleculeOptimization",
    "MoleculeOptimizations",
    # Critic Outputs
    "CriticMoleculeEvaluation",
    "CriticEvaluations",
    # Node 1: Info Extraction
    "ExtractInitialInfo",
    # Node 3: Selection
    "SelectPromisingMolecule",
    "SharedMoleculeAnalysis",
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
    # Trace utils
    "safe_serialize",
    "serialize_message",
    "build_invocation_trace",
    "format_trace_block",
]
