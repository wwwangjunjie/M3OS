#!/usr/bin/env python3
"""
Improved MCP Server for M3OS Drug Discovery Tools
Based on MCP Chinese Getting Started Guide best practices
Uses FastMCP for better structure and maintainability
"""

import asyncio
import argparse
import json
import os
import sys
import signal
from typing import Any, Dict, List, Optional, Union
import logging

import ast
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics.pairwise import cosine_similarity
# Add M3OS to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use FastMCP as recommended in the guide
from mcp.server import FastMCP


from m3os.agents_v5.tools.retrieval import (
    download_relevant_papers,
    fetch_uniprot_fasta,
    get_drug_smiles,
    get_uniprot_ids,
    question_answering,
    search,
)
from m3os.agents_v5.tools.molecule_images import (
    generate_molecule_difference_image as generate_molecule_difference_image_file,
    generate_molecule_pair_difference_image as generate_molecule_pair_difference_image_file,
    generate_molecule_topology_image as generate_molecule_topology_image_file,
)
from m3os.agents_v5.tools.molecular_similarity import (
    molecular_similarity_search as molecular_similarity_search_impl,
)
from m3os.agents_v5.services.smiles_validation import (
    validate_smiles_dataframe,
    validate_smiles,
    validate_smiles_batch_messages,
)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server with descriptive name
app = FastMCP('m3os-drug-discovery')

# =============================================================================
# RETRIEVAL TOOLS
# =============================================================================

@app.tool()
async def web_search(query: str) -> str:
    """
    Search for information relevant to drug discovery.
    
    Args:
        query: Search query.
        
    Returns:
        Summary of search results.
    """
    return str(search(query))


@app.tool()
async def get_protein_uniprot_ids(
    protein_name: str,
    reviewed: bool | None = None,
    organism_id: str | None = None,
) -> str:
    """
    Retrieve UniProt IDs for a protein name.
    
    Args:
        protein_name: Protein name.
        reviewed: Whether to restrict results to reviewed UniProtKB entries.
            None prefers reviewed entries and relaxes the filter if necessary;
            True strictly requires reviewed entries; False applies no filter.
        organism_id: Optional NCBI taxonomy ID used to restrict the organism.
            Examples: human/Homo sapiens=9606, mouse/Mus musculus=10090,
            rat/Rattus norvegicus=10116, dog/Canis lupus familiaris=9615,
            monkey/Macaca mulatta=9544, E. coli=562,
            yeast/Saccharomyces cerevisiae=4932, zebrafish/Danio rerio=7955,
            SARS-CoV-2=2697049, HIV-1=11676.
            Supply this only when the organism is specified and its taxonomy
            ID is known. Without it, human (9606) is preferred and the search
            may broaden to all organisms if needed.
        
    Returns:
        UniProt IDs and descriptive metadata.
    """
    return str(
        get_uniprot_ids(
            protein_name,
            reviewed=reviewed,
            organism_id=organism_id,
        )
    )


@app.tool()
async def get_protein_sequence(uniprot_id: str) -> str:
    """
    Retrieve a protein sequence by UniProt ID.
    
    Args:
        uniprot_id: UniProt database ID.
        
    Returns:
        Protein sequence in FASTA format.
    """
    return str(fetch_uniprot_fasta(uniprot_id))


@app.tool()
async def get_molecule_smiles(drug_name: str) -> str:
    """
    Retrieve a molecule's SMILES from ChEMBL.
    
    Args:
        drug_name: Molecule or drug name.
        
    Returns:
        SMILES string and candidate-molecule information.
    """
    return str(get_drug_smiles(drug_name))


@app.tool()
async def answer_research_paper_question(query: str) -> str:
    """
    Answer a question using downloaded research papers and RAG.

    Args:
        query: Research question.

    Returns:
        Answer grounded in the paper content.
    """
    return str(question_answering(query))


@app.tool()
async def download_research_papers(query: str, max_papers: int = 3) -> str:
    """
    Download relevant academic papers.

    Args:
        query: Research-topic query.
        max_papers: Maximum number of open-access papers; defaults to 3 and
            is capped at 10.

    Returns:
        Download status and paper metadata.
    """
    return str(download_relevant_papers(query, max_papers=max_papers))


@app.tool()
async def molecular_similarity_search(smiles_a: str, smiles_b: str) -> str:
    """
    Compute molecule or scaffold similarity.

    Calculate RDKit Morgan-fingerprint Tanimoto similarity for two SMILES.

    Args:
        smiles_a: First molecule or scaffold SMILES.
        smiles_b: Second molecule or scaffold SMILES.

    Returns:
        JSON string with Tanimoto similarity and fingerprint settings.
    """
    return molecular_similarity_search_impl(smiles_a=smiles_a, smiles_b=smiles_b)


# =============================================================================
# PREDICTION TOOLS
# =============================================================================

@app.tool()
async def validate_smiles_string(smiles: str) -> str:
    """
    Validate the chemical structure of a SMILES string.
    
    Args:
        smiles: SMILES string to validate.
        
    Returns:
        Validation result and details.
    """
    is_valid, canonical_smiles, error = validate_smiles(smiles)
    if is_valid:
        canonical_note = f"; canonical SMILES: {canonical_smiles}" if canonical_smiles else ""
        return f"SMILES '{smiles}' is a valid chemical structure{canonical_note}"
    else:
        return f"SMILES '{smiles}' is not a valid chemical structure; error: {error or 'RDKit could not parse SMILES'}"


@app.tool()
async def validate_smiles_batch(smiles_list: List[str]) -> Dict[str, str]:
    """
    Validate the chemical structures of multiple SMILES strings.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        Mapping from input SMILES to "valid" or a specific error message.
    """
    return validate_smiles_batch_messages(smiles_list)


@app.tool()
async def clean_smiles_dataset(csv_path: str) -> str:
    """
    Remove invalid SMILES strings from a dataset.
    
    Args:
        csv_path: Path to the input CSV file.
        
    Returns:
        Cleaning statistics.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    original_count = len(df)
    cleaned_df = validate_smiles_dataframe(df)
    cleaned_count = len(cleaned_df)
    cleaned_df.to_csv(csv_path, index=False)
    
    return f"Dataset cleaning complete: {original_count} original records, {cleaned_count} valid records retained"



@app.tool()
async def smiles_to_fragments_str(smiles_list: list) -> dict:
    """
    Fragment multiple molecules' SMILES strings into functional groups and return a dictionary of results.
    
    Args:
        smiles_list (list): List of input molecular SMILES strings
        
    Returns:
        dict: Dictionary where key is the SMILES string and value is the string representation of its functional group list.
              Returns a dictionary with error messages if an error occurs during processing of any SMILES.
    """
    from m3os.agents_v5.tools.fragments import smiles2fragments_str

    return smiles2fragments_str(smiles_list=smiles_list)


@app.tool()
async def generate_molecule_topology_image(
    smiles: str,
    output_dir: Optional[str] = None,
    image_prefix: str = "molecule_topology",
    width: int = 520,
    height: int = 420,
    show_atom_indices: bool = False,
) -> str:
    """
    Generate a molecule 2D topology PNG from SMILES and return the saved image path.

    Args:
        smiles: Molecule SMILES.
        output_dir: Optional directory for the PNG. Defaults to tmp/molecule_images.
        image_prefix: Filename prefix for the generated image.
        width: Image width in pixels.
        height: Image height in pixels.
        show_atom_indices: Whether to annotate atoms with zero-based RDKit atom indices.

    Returns:
        JSON string containing image_path, relative_image_path, canonical_smiles, and method.
    """
    try:
        result = generate_molecule_topology_image_file(
            smiles=smiles,
            output_dir=output_dir,
            image_prefix=image_prefix,
            width=width,
            height=height,
            show_atom_indices=show_atom_indices,
        )
    except Exception as exc:
        result = {"valid": False, "image_path": "", "error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(result, ensure_ascii=False)


@app.tool()
async def generate_molecule_difference_image(
    smiles: str,
    reference_smiles: str,
    output_dir: Optional[str] = None,
    image_prefix: str = "molecule_difference",
    width: int = 620,
    height: int = 460,
    mcs_timeout: int = 20,
    show_atom_indices: bool = False,
) -> str:
    """
    Generate one molecule 2D topology PNG with differences highlighted against another molecule.

    The generated image shows only `smiles`; green marks the MCS shared with
    `reference_smiles`, orange marks atoms/bonds different from the reference,
    and purple marks stereochemical differences when detected.

    Args:
        smiles: Molecule SMILES to draw.
        reference_smiles: Comparison/reference molecule SMILES.
        output_dir: Optional directory for the PNG. Defaults to tmp/molecule_images.
        image_prefix: Filename prefix for the generated image.
        width: Image width in pixels.
        height: Image height in pixels.
        mcs_timeout: RDKit MCS timeout in seconds.
        show_atom_indices: Whether to annotate atoms with zero-based RDKit atom indices.

    Returns:
        JSON string containing image_path, relative_image_path, MCS summary, and method.
    """
    try:
        result = generate_molecule_difference_image_file(
            smiles=smiles,
            reference_smiles=reference_smiles,
            output_dir=output_dir,
            image_prefix=image_prefix,
            width=width,
            height=height,
            mcs_timeout=mcs_timeout,
            show_atom_indices=show_atom_indices,
        )
    except Exception as exc:
        result = {"valid": False, "image_path": "", "error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(result, ensure_ascii=False)


@app.tool()
async def generate_molecule_pair_difference_image(
    before_smiles: str,
    after_smiles: str,
    output_dir: Optional[str] = None,
    image_prefix: str = "molecule_pair_difference",
    width: int = 1240,
    height: int = 460,
    mcs_timeout: int = 20,
    show_atom_indices: bool = False,
) -> str:
    """
    Generate a paired before/after molecule difference PNG and return the saved image path.

    The left side is the original/pre-optimization molecule and the right side
    is the changed/post-optimization molecule. Green marks the shared MCS,
    orange marks changed/added/removed regions, and purple marks stereochemical
    differences when detected.

    Args:
        before_smiles: Original/pre-optimization molecule SMILES.
        after_smiles: Changed/post-optimization molecule SMILES.
        output_dir: Optional directory for the PNG. Defaults to tmp/molecule_images.
        image_prefix: Filename prefix for the generated image.
        width: Total image width target in pixels.
        height: Per-molecule panel height target in pixels.
        mcs_timeout: RDKit MCS timeout in seconds.
        show_atom_indices: Whether to annotate atoms with zero-based RDKit atom indices.

    Returns:
        JSON string containing image_path, relative_image_path, MCS summary, and method.
    """
    try:
        result = generate_molecule_pair_difference_image_file(
            before_smiles=before_smiles,
            after_smiles=after_smiles,
            output_dir=output_dir,
            image_prefix=image_prefix,
            width=width,
            height=height,
            mcs_timeout=mcs_timeout,
            show_atom_indices=show_atom_indices,
        )
    except Exception as exc:
        result = {"valid": False, "image_path": "", "error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(result, ensure_ascii=False)



# =============================================================================
# MAIN SERVER EXECUTION
# =============================================================================

def parse_server_args() -> argparse.Namespace:
    """Parse transport settings for stdio compatibility or resident HTTP mode."""
    parser = argparse.ArgumentParser(description="M3OS main MCP server")
    default_transport = os.getenv("MCP_SERVER_TRANSPORT", "stdio").replace("_", "-")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default=default_transport,
    )
    parser.add_argument("--host", default=os.getenv("MCP_SERVER_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_SERVER_PORT", "8010")),
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info("Shutting down M3OS MCP server...")
    sys.exit(0)

if __name__ == "__main__":
    server_args = parse_server_args()
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info(
            "Starting M3OS MCP Server with FastMCP transport=%s host=%s port=%s...",
            server_args.transport,
            server_args.host,
            server_args.port,
        )
        logger.info("Available tools: retrieval and molecular optimization tools")

        app.settings.host = server_args.host
        app.settings.port = server_args.port
        app.run(transport=server_args.transport)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal server error: {e}")
        sys.exit(1)
