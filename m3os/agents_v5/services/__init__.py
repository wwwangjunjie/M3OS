"""Service exports for M3OS.

This module provides business logic services that support the workflow nodes
and agents, including LLM management, MCP client handling, knowledge retrieval,
and molecular utilities.
"""

from m3os.agents_v5.services.llm_factory import (
    LLMFactory,
    LLMProvider,
    ClaudeProvider,
    GeminiProvider,
    KimiProvider,
    create_base_model,
    get_middleware,
)
from m3os.agents_v5.services.mcp_client import (
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
)
from m3os.agents_v5.services.document_rag import (
    DocumentRAGService,
    DocumentIngestionResult,
)
from m3os.agents_v5.services.knowledge_context import (
    KnowledgeContextService,
)
from m3os.agents_v5.services.m3os_wiki import (
    M3OSWikiKnowledgeService,
)
from m3os.agents_v5.services.molecule_utils import (
    MoleculeUtils,
)
from m3os.agents_v5.services.task_preparation import (
    TaskPreparationService,
)
from m3os.agents_v5.services.admet_property_selection import (
    ADMETPropertySelectionService,
)
from m3os.agents_v5.services.table_context import (
    TableSource,
    build_table_context,
    combine_context_blocks,
    read_table_files,
)
from m3os.agents_v5.services.structure_files import (
    PDBSequenceRecord,
    StructureFileResult,
    build_structure_file_context,
    collect_structure_files,
    pdb_file_to_protein_sequence,
    process_uploaded_structure_files,
    read_pdb,
    read_sdf,
    sdf_file_to_smiles,
)

__all__ = [
    # LLM Factory
    "LLMFactory",
    "LLMProvider",
    "ClaudeProvider",
    "GeminiProvider",
    "KimiProvider",
    "create_base_model",
    "get_middleware",
    # MCP Client
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
    # Molecule Utils
    "MoleculeUtils",
    # Task Preparation
    "TaskPreparationService",
    "ADMETPropertySelectionService",
    # Table Context
    "TableSource",
    "build_table_context",
    "combine_context_blocks",
    "read_table_files",
    # Structure files
    "PDBSequenceRecord",
    "StructureFileResult",
    "build_structure_file_context",
    "collect_structure_files",
    "pdb_file_to_protein_sequence",
    "process_uploaded_structure_files",
    "read_pdb",
    "read_sdf",
    "sdf_file_to_smiles",
]
