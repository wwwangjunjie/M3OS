"""Molecule utilities service for common molecular operations.

This module provides utility functions for working with molecules,
including fragment extraction, IUPAC name generation, and SMILES manipulation.
"""

import json
from typing import Dict, List, Optional, Any

from m3os.agents_v5.services.mcp_client import MCPClientManager


class MoleculeUtils:
    """Utility class for molecule-related operations.
    
    This class provides methods for common operations like getting
    molecular fragments, IUPAC names, and merging molecular information.
    """
    
    def __init__(self, mcp_client: MCPClientManager):
        """Initialize molecule utilities.
        
        Args:
            mcp_client: MCP client manager for accessing molecule tools
        """
        self.mcp_client = mcp_client
    
    async def get_fragments(self, smiles: str) -> Dict[str, Any]:
        """Get functional groups/fragments for a molecule.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            Dictionary mapping SMILES to fragment information
        """
        try:
            result = await self.mcp_client.invoke_tool(
                tool_name="smiles_to_fragments_str",
                params={"smiles_list": [smiles]}
            )
            return result
        except Exception as e:
            print(f"Error fragmenting molecule: {e}")
            return {}
    
    async def get_fragments_batch(self, smiles_list: List[str]) -> Dict[str, Any]:
        """Get fragments for multiple molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary mapping SMILES to fragment information
        """
        try:
            result = await self.mcp_client.invoke_tool(
                tool_name="smiles_to_fragments_str",
                params={"smiles_list": smiles_list}
            )
            return result
        except Exception as e:
            print(f"Error fragmenting molecules: {e}")
            return {}
    
    async def get_iupac_name(self, smiles: str) -> str:
        """Get IUPAC name for a molecule.
        
        Args:
            smiles: SMILES string of the molecule
            
        Returns:
            IUPAC name string, or "None" on error
        """
        try:
            result = await self.mcp_client.invoke_tool(
                tool_name="generate_iupac_name",
                params={"smiles_list": [smiles]},
                server_hint="mcp_server_iupac_gen"
            )
            # Result is a dict mapping SMILES to IUPAC name
            if isinstance(result, dict) and smiles in result:
                return result[smiles]
            return str(result)
        except Exception as e:
            print(f"Error generating IUPAC: {e}")
            return "None"
    
    async def get_iupac_names_batch(self, smiles_list: List[str]) -> Dict[str, str]:
        """Get IUPAC names for multiple molecules.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Dictionary mapping SMILES to IUPAC names
        """
        print("[smiles_list in get_iupac_names_batch]:", smiles_list)
        try:
            result = await self.mcp_client.invoke_tool(
                tool_name="generate_iupac_name",
                params={"smiles_list": smiles_list},
                server_hint="mcp_server_iupac_gen"
            )
            if isinstance(result, dict):
                return result
            return result
        except Exception as e:
            print(f"Error generating IUPAC names: {e}")
            return {}
    
    @staticmethod
    def merge_fragments_and_iupac(
        fragments_info: str,
        iupac_info: str
    ) -> List[Dict[str, str]]:
        """Merge fragments and IUPAC information into a list.
        
        Args:
            fragments_info: JSON string of fragments dict {smiles: fragments}
            iupac_info: JSON string of IUPAC dict {smiles: iupac_name}
            
        Returns:
            List of dictionaries with SMILES, Functional Groups, and IUPAC Name
            
        Raises:
            ValueError: If JSON parsing fails or keys don't match
        """
        # Parse JSON strings
        try:
            fragments_dict = json.loads(fragments_info)
            iupac_dict = json.loads(iupac_info)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}")
        
        # Validate keys match
        fragments_keys = set(fragments_dict.keys())
        iupac_keys = set(iupac_dict.keys())
        
        if fragments_keys != iupac_keys:
            missing_in_iupac = fragments_keys - iupac_keys
            missing_in_fragments = iupac_keys - fragments_keys
            raise ValueError(
                f"Key mismatch detected.\n"
                f"Missing in iupac_info: {missing_in_iupac}\n"
                f"Missing in fragments_info: {missing_in_fragments}"
            )
        
        # Merge into list format
        merged_list = []
        for smiles in fragments_keys:
            merged_list.append({
                "SMILES": smiles,
                "Functional Groups": fragments_dict[smiles],
                "IUPAC Name": iupac_dict[smiles]
            })
        
        return merged_list
    
    async def get_molecule_info(
        self,
        smiles: str
    ) -> Dict[str, Any]:
        """Get complete information for a molecule.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with fragments, IUPAC name, and SMILES
        """
        fragments = await self.get_fragments(smiles)
        iupac = await self.get_iupac_name(smiles)
        
        return {
            "smiles": smiles,
            "fragments": fragments.get(smiles, "None") if isinstance(fragments, dict) else "None",
            "iupac": iupac if iupac != "None" else "None"
        }
