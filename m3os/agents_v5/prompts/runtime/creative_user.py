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
This prompt is for one MCGS expansion round only. Any candidate count embedded in the Optimization Goal or Project Manager Brief is the MCGS cumulative target across rounds, not the number you should produce now. For a target-activity objective, return the complete Nesso or intersection-filtered set of at most 10 final molecules when that set is non-empty. If the intersection is empty but the tool returns ADMET/hard-qualified molecules with activity measurements, use that evidence to propose a compact batch of unverified next-step analogs for Critic evaluation; do not claim that they satisfy the activity target. If no molecule passes the ADMET/hard stage, return no Creative molecule for that route. For a non-activity objective, return the complete final filtered set, normally 5 molecules. Do not add molecules merely to fill the user's total requested count.

## [SHARED CURRENT ANALYSIS CONTEXT]
* **Shared Analysis Summary:** {shared_analysis_summary}
* **Fragments To Keep:** {shared_keep_fragments}
* **Modifiable Fragments:** {shared_modifiable_fragments}
* **Risk Alerts:** {shared_risk_alerts}
* **Priority Directions:** {shared_priority_directions}

## [CREATIVE EXPLORER AUDITED MEMORY]
{agent_memory}

## [SCREENING CONTEXT HISTORY]
{screening_context_history}

[Note]: Be terse. Preserve only core design intent, edit location, and task-relevant property reason.
[Note]: Answer in concise language.
"""
