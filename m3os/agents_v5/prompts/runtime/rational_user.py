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

## [MCGS ROUND CONTEXT]

- Current expansion round: {current_mcgs_round}

## [CURRENT ROUND CANDIDATE BUDGET]
This prompt is for one MCGS expansion round only. Any candidate count embedded in the Optimization Goal or Project Manager Brief is the MCGS cumulative target across rounds, not the number you should produce now. Return only a compact current-round batch of 5-7 final molecules, never more than 7. Do not hand-author extra molecules to fill the user's total requested count.

## [SHARED CURRENT ANALYSIS CONTEXT]
* **Shared Analysis Summary:** {shared_analysis_summary}
* **Fragments To Keep:** {shared_keep_fragments}
* **Modifiable Fragments:** {shared_modifiable_fragments}
* **Risk Alerts:** {shared_risk_alerts}
* **Priority Directions:** {shared_priority_directions}

## [RATIONAL DESIGNER AUDITED MEMORY]
{agent_memory}

## [SCREENING CONTEXT HISTORY]
{screening_context_history}

[Note]: Be concise but preserve the core strategy synthesis: medicinal chemistry knowledge used, case-derived transformation signal, edit location, and task-relevant property reason.
"""
