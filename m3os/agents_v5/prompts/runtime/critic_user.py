USER_PROMPTS = """
## [Current Task Data]

* **The SMILES of Initial Molecule to be optimized (this is the molecule initially submitted by the user for optimization, which serves as the root node in the Monte Carlo search graph):** {initial_smiles}
* **The IUPAC Name of Initial Molecule to be optimized:** {initial_iupac}
* **The Functional Groups of Initial Molecule to be optimized:** {initial_fragments}
* **The protein sequence corresponding to the initial molecule to be optimized:** {initial_protein_squence}
* **Optimization Goal:** {optimization_goal}
* **Project Manager Brief:** {project_manager_brief}
* **The SMILES of Current Molecule to be optimized (during the Monte Carlo graph search process, this is the molecule currently identified for optimization.):** {current_smiles}
* **The IUPAC Name of Current Molecule to be optimized:** {current_iupac}
* **The Functional Groups of Current Molecule to be optimized:** {current_fragments}
* **Current Selection Context:** {current_selection_context}
* **Optimized Molecule Candidates for the Current Molecule (these are the molecules you need to evaluate currently):** {optimized_molecules}

## [SHARED CURRENT ANALYSIS CONTEXT]
* **Shared Analysis Summary:** {shared_analysis_summary}
* **Fragments To Keep:** {shared_keep_fragments}
* **Modifiable Fragments:** {shared_modifiable_fragments}
* **Risk Alerts:** {shared_risk_alerts}
* **Priority Directions:** {shared_priority_directions}

## [CRITIC AUDITED MEMORY]
{agent_memory}

[Note]: Be terse. Preserve only score-relevant evidence, key risk, and task alignment.
[Note]: Answer in concise language.
"""
