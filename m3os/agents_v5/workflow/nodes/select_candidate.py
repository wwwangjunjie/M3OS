"""Node 3: Select Best Candidate

This node retrieves the top-5 nodes from the graph and uses LLM reasoning
to select the most promising molecule for further optimization.
"""

import json
from typing import Any, Dict, List

from m3os.agents_v5.core.state import MCGSState
from m3os.agents_v5.core.retry_utils import (
    invoke_with_retry,
    validate_structured_response,
    LLMInvocationError,
    InvalidResponseError,
)
from m3os.agents_v5.core.models import SelectPromisingMolecule
from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.services.llm_factory import create_base_model
from m3os.agents_v5.services.mcp_client import MCPClientManager
from m3os.agents_v5.services.molecule_utils import MoleculeUtils
from m3os.agents_v5.services.smiles_validation import validate_smiles
from m3os.agents_v5.services.molecular_context import (
    without_explicit_molecular_auxiliary_fields,
)
from m3os.agents_v5.tools.mcgs_tools import mcgs_get_top_5_nodes
from m3os.agents_v5.prompts import (
    SELECT_CANDIDATE_SYSTEM_PROMPT,
    SELECT_CANDIDATE_USER_PROMPT,
)


class SelectBestCandidateNode:
    """Node for selecting the best candidate molecule.
    
    This node:
    1. Gets the top-5 nodes from the graph based on UCT values
    2. Retrieves fragments and IUPAC names for candidates
    3. Uses LLM to select the most promising molecule
    4. Returns the selected molecule's information
    """
    
    def __init__(
        self,
        config: AgentConfig,
        mcp_client: MCPClientManager,
    ):
        """Initialize the node.
        
        Args:
            config: Agent configuration
            mcp_client: MCP client manager
        """
        self.config = config
        self.mcp_client = mcp_client
        self.molecule_utils = MoleculeUtils(mcp_client)
    
    async def __call__(self, state: MCGSState) -> Dict[str, Any]:
        """Execute the node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary of updates to the state
        """
        print("\n=== Node: select_best_candidate ===")
        
        mol_graph = state["mcgs_graph"]
        
        # Get top-5 nodes from graph
        top_5, mol_graph = mcgs_get_top_5_nodes(mol_graph=mol_graph)

        # print("[top_5]:", top_5)
        facts_cache = dict(state.get("molecule_facts_cache") or {})
        if not self.config.enable_molecular_auxiliary_context:
            for smiles in top_5:
                key = self._canonical_smiles_key(smiles)
                if key:
                    facts_cache[key] = {
                        "smiles": smiles,
                        "canonical_smiles": key,
                        "iupac": "",
                        "fragments": "",
                    }
            candidates_info = self._smiles_only_candidate_info(top_5, mol_graph)
            select_result = await self._select_candidate(state, candidates_info)
            current_smiles, invalid_selection = self._resolve_selected_candidate(
                select_result.smiles,
                top_5,
            )
            current_select_reason = select_result.reason
            if invalid_selection:
                current_select_reason = (
                    f"{current_select_reason}\n\n"
                    f"Selection fallback: {invalid_selection['reason']} "
                    f"Using graph candidate {current_smiles}."
                )
            print("Molecular auxiliary context disabled: selecting from SMILES-only candidates.")
            print("\n===[END] Node: select_best_candidate ===")
            return {
                "current_info_list": [
                    {
                        "current_smiles": current_smiles,
                        "current_iupac": "",
                        "current_fragments": "",
                        "current_selection_context": select_result.selection_context,
                        "current_select_reason": current_select_reason,
                    }
                ],
                "molecule_facts_cache": facts_cache,
                "mcgs_graph": mol_graph,
                "invalid_candidate_selection": invalid_selection,
            }

        top_5_fragments: Dict[str, Any] = {}
        top_5_iupac: Dict[str, Any] = {}
        missing_fragments: List[str] = []
        missing_iupac: List[str] = []

        for smiles in top_5:
            key = self._canonical_smiles_key(smiles)
            facts = facts_cache.get(key, {})
            if isinstance(facts, dict):
                if facts.get("fragments"):
                    top_5_fragments[smiles] = facts.get("fragments")
                else:
                    missing_fragments.append(smiles)
                if facts.get("iupac"):
                    top_5_iupac[smiles] = facts.get("iupac")
                else:
                    missing_iupac.append(smiles)
            else:
                missing_fragments.append(smiles)
                missing_iupac.append(smiles)
        
        # Get fragments for candidates
        try:
            fetched_fragments = (
                await self.molecule_utils.get_fragments_batch(missing_fragments)
                if missing_fragments
                else {}
            )
            fetched_fragments = self._normalize_mapping(fetched_fragments)
            top_5_fragments.update(fetched_fragments)
            print(f"Fragments: {top_5_fragments}")
        except Exception as e:
            print(f"Error fragmenting molecules: {e}")
            fetched_fragments = {}

        # Ensure top_5_fragments is a dict (handle JSON string case)
        top_5_fragments = self._normalize_mapping(top_5_fragments)

        # Get IUPAC names for candidates
        try:
            fetched_iupac = (
                await self.molecule_utils.get_iupac_names_batch(missing_iupac)
                if missing_iupac
                else {}
            )
            fetched_iupac = self._normalize_mapping(fetched_iupac)
            top_5_iupac.update(fetched_iupac)
            print(f"IUPAC: {top_5_iupac}")
        except Exception as e:
            print(f"Error generating IUPAC names: {e}")
            fetched_iupac = {}

        # Ensure top_5_iupac is a dict (handle JSON string case)
        top_5_iupac = self._normalize_mapping(top_5_iupac)

        for smiles in top_5:
            key = self._canonical_smiles_key(smiles)
            if not key:
                continue
            facts = dict(facts_cache.get(key, {})) if isinstance(facts_cache.get(key), dict) else {}
            facts.update(
                {
                    "smiles": smiles,
                    "canonical_smiles": key,
                    "iupac": top_5_iupac.get(smiles, facts.get("iupac")),
                    "fragments": top_5_fragments.get(smiles, facts.get("fragments")),
                }
            )
            facts_cache[key] = facts
        
        # Merge fragments and IUPAC info
        candidates_info = self._merge_info(top_5_fragments, top_5_iupac, mol_graph=mol_graph)
        
        # Select best candidate with LLM
        select_result = await self._select_candidate(state, candidates_info)
        
        current_smiles, invalid_selection = self._resolve_selected_candidate(
            select_result.smiles,
            top_5,
        )
        current_selection_context = select_result.selection_context
        current_select_reason = select_result.reason
        if invalid_selection:
            current_select_reason = (
                f"{current_select_reason}\n\n"
                f"Selection fallback: {invalid_selection['reason']} "
                f"Using graph candidate {current_smiles}."
            )
        
        # Get fragments and IUPAC for selected molecule
        current_key = self._canonical_smiles_key(current_smiles)
        current_facts = facts_cache.get(current_key, {}) if current_key else {}
        current_fragments = top_5_fragments.get(current_smiles) or current_facts.get("fragments") or "None"
        current_iupac = top_5_iupac.get(current_smiles) or current_facts.get("iupac") or "None"
        
        print("\n===[END] Node: select_best_candidate ===")
        
        # Return updates
        return {
            "current_info_list": [
                {
                    "current_smiles": current_smiles,
                    "current_iupac": current_iupac,
                    "current_fragments": current_fragments,
                    "current_selection_context": current_selection_context,
                    "current_select_reason": current_select_reason,
                }
            ],
            "molecule_facts_cache": facts_cache,
            "mcgs_graph": mol_graph,
            "invalid_candidate_selection": invalid_selection,
        }

    @staticmethod
    def _normalize_mapping(value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                print("Warning: Failed to parse molecule info mapping as JSON")
                value = {}
        if not isinstance(value, dict):
            print(f"Warning: molecule info mapping is not a dict (type: {type(value)})")
            return {}
        return value

    @staticmethod
    def _canonical_smiles_key(smiles: Any) -> str:
        ok, canonical, _error = validate_smiles(smiles)
        return canonical if ok and canonical else str(smiles or "").strip()

    def _resolve_selected_candidate(
        self,
        selected_smiles: Any,
        top_candidates: List[str],
    ) -> tuple[str, Dict[str, Any] | None]:
        """Keep LLM selection inside the valid graph candidate set."""
        candidate_by_canonical: Dict[str, str] = {}
        first_valid = ""
        for candidate in top_candidates:
            ok, canonical, _error = validate_smiles(candidate)
            if not ok or not canonical:
                continue
            candidate_text = str(candidate or "").strip()
            candidate_by_canonical[canonical] = candidate_text
            if not first_valid:
                first_valid = candidate_text

        selected_text = str(selected_smiles or "").strip()
        ok, selected_canonical, selected_error = validate_smiles(selected_text)
        if ok and selected_canonical in candidate_by_canonical:
            return candidate_by_canonical[selected_canonical], None

        fallback = first_valid or (str(top_candidates[0]).strip() if top_candidates else selected_text)
        reason = (
            f"selected SMILES {selected_text!r} is invalid: {selected_error}"
            if not ok
            else f"selected SMILES {selected_text!r} is not in the graph candidate set"
        )
        return fallback, {
            "selected_smiles": selected_text,
            "fallback_smiles": fallback,
            "reason": reason,
        }
    
    def _merge_info(
        self,
        fragments_dict: Dict[str, Any],
        iupac_dict: Dict[str, str],
        mol_graph: Any = None,
    ) -> List[Dict[str, Any]]:
        """Merge fragments and IUPAC information.

        Args:
            fragments_dict: Dictionary mapping SMILES to fragments (or JSON string)
            iupac_dict: Dictionary mapping SMILES to IUPAC names (or JSON string)

        Returns:
            List of merged dictionaries
        """
        # Handle string inputs (JSON strings from MCP tool)
        if isinstance(fragments_dict, str):
            try:
                fragments_dict = json.loads(fragments_dict)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse fragments_dict as JSON: {fragments_dict[:100]}...")
                fragments_dict = {}

        if isinstance(iupac_dict, str):
            try:
                iupac_dict = json.loads(iupac_dict)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse iupac_dict as JSON: {iupac_dict[:100]}...")
                iupac_dict = {}

        # Ensure dict types
        if not isinstance(fragments_dict, dict):
            print(f"Warning: fragments_dict is not a dict (type: {type(fragments_dict)}), using empty dict")
            fragments_dict = {}
        if not isinstance(iupac_dict, dict):
            print(f"Warning: iupac_dict is not a dict (type: {type(iupac_dict)}), using empty dict")
            iupac_dict = {}

        node_meta = self._node_metadata_by_smiles(mol_graph)
        merged = []
        for smiles in fragments_dict.keys():
            metadata = node_meta.get(smiles, {})
            merged.append({
                "SMILES": smiles,
                "Functional Groups": fragments_dict[smiles],
                "IUPAC Name": iupac_dict.get(smiles, "Unknown"),
                "Critic Score": metadata.get("critic_score"),
                "Root Similarity": metadata.get("root_similarity"),
                "MCGS Selection Score": metadata.get("mcgs_selection_score"),
                "Graph UCT": metadata.get("uct_value"),
                "Iteration": metadata.get("iteration"),
            })
        return merged

    def _node_metadata_by_smiles(self, mol_graph: Any) -> Dict[str, Dict[str, Any]]:
        if mol_graph is None or not hasattr(mol_graph, "get_all_nodes"):
            return {}
        metadata: Dict[str, Dict[str, Any]] = {}
        try:
            nodes = list(mol_graph.get_all_nodes())
        except Exception:
            return metadata
        for node in nodes:
            smiles = str(getattr(node, "smiles", "") or "")
            if not smiles:
                continue
            properties = getattr(node, "properties", {}) or {}
            if not isinstance(properties, dict):
                properties = {}
            score = getattr(node, "intrinsic_score", None)
            metadata[smiles] = {
                "critic_score": properties.get("critic_score", score),
                "root_similarity": properties.get("root_similarity"),
                "mcgs_selection_score": properties.get("mcgs_selection_score", score),
                "uct_value": getattr(node, "uct_value", None),
                "iteration": getattr(node, "iteration", None),
            }
        return metadata

    def _smiles_only_candidate_info(
        self,
        smiles_list: List[str],
        mol_graph: Any,
    ) -> List[Dict[str, Any]]:
        node_meta = self._node_metadata_by_smiles(mol_graph)
        return [
            {
                "SMILES": smiles,
                "Critic Score": node_meta.get(smiles, {}).get("critic_score"),
                "Root Similarity": node_meta.get(smiles, {}).get("root_similarity"),
                "MCGS Selection Score": node_meta.get(smiles, {}).get("mcgs_selection_score"),
                "Graph UCT": node_meta.get(smiles, {}).get("uct_value"),
                "Iteration": node_meta.get(smiles, {}).get("iteration"),
            }
            for smiles in smiles_list
        ]
    
    async def _select_candidate(
        self,
        state: MCGSState,
        candidates_info: List[Dict[str, str]]
    ) -> SelectPromisingMolecule:
        """Use LLM to select the best candidate.
        
        Args:
            state: Current workflow state
            candidates_info: List of candidate molecule information
            
        Returns:
            Selection result
        """
        llm = create_base_model(self.config.llm)
        structured_llm = llm.with_structured_output(SelectPromisingMolecule, include_raw=True)
        
        # Get current info from state
        current_info = state["current_info_list"][-1] if state["current_info_list"] else {}
        
        # Build messages for candidate selection.
        system_prompt = (
            SELECT_CANDIDATE_SYSTEM_PROMPT
            + "\n\nSimilarity retention rule: candidate metadata may include Root Similarity "
            "and MCGS Selection Score. Treat similarity to the initial/root molecule as a "
            "soft but important constraint; prefer candidates that balance optimization "
            "benefit with pharmacophore/scaffold continuity, and avoid selecting a highly "
            "drifted molecule unless the rationale clearly justifies the tradeoff."
        )
        user_prompt = SELECT_CANDIDATE_USER_PROMPT.format(
            initial_smiles=state["initial_smiles"],
            initial_iupac=state["initial_iupac"],
            initial_fragments=state["initial_fragments"],
            optimization_goal=state["optimization_goal"],
            project_manager_brief=state.get("project_manager_brief", ""),
            candidates_info=candidates_info,
        )
        if not self.config.enable_molecular_auxiliary_context:
            system_prompt = without_explicit_molecular_auxiliary_fields(system_prompt)
            user_prompt = without_explicit_molecular_auxiliary_fields(user_prompt)

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    }
                ]
            }
        ]
        
        def _invoke():
            return structured_llm.invoke(messages)
        
        response = await invoke_with_retry(
            _invoke,
            max_retries=3,
            base_delay=1.0,
            validate_fn=validate_structured_response,
            operation_name="Select candidate LLM call"
        )
        
        parsed = response["parsed"]
        raw_msg = response["raw"]
        
        print("[select_response]:", parsed)
        print("[select best candidate TOKEN]:", raw_msg.usage_metadata)
        
        return parsed
