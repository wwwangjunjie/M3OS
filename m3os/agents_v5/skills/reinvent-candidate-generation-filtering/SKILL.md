---
name: reinvent-candidate-generation-filtering
description: Creative Explorer workflow for turning shared analysis, medicinal chemistry guidance, uploaded-document cues, and already-provided case evidence into focused REINVENT candidate generation inputs, then selecting the appropriate ADMET, Nesso, or constraint-aware final filter.
---

# reinvent-candidate-generation-filtering

## Overview

This skill defines the end-to-end workflow for controlled generative exploration around a current molecule. Its default non-activity path is LibInvent/R-group generation followed by filtering, because that workflow identifies an editable molecular site, generates replacements at that site, and supports directional structure-property optimization. Target-activity optimization uses similarity-constrained Mol2Mol generation. It also covers LinkInvent linker generation through the external REINVENT MCP service for explicitly requested linker moves. Use the REINVENT MCP tools directly.

This skill produces a compact candidate batch for one MCGS expansion round only. Any user-requested total candidate count is handled by the graph search across rounds; do not fill that total here.

## Instructions

1. Confirm evidence needed for generation
- Start from shared analysis, user constraints, frozen ADMET policy, role memory, lightweight medicinal chemistry answer memory, and any already-provided rational case evidence. Do not re-expand complete molecule diagnosis if shared analysis is already provided; only output incremental judgments.
- Before choosing templates, filters, or broad exploration tactics, reuse available medicinal chemistry memory. If it does not cover the exact scaffold, property, risk, uploaded-document/table cue, or generation decision, call `ask_medchem_knowledge` with a focused question that includes the current scaffold, objective, risk, and decision you need.
- If uploaded paper or table evidence could affect template choice, filtering, SAR, ADMET, or scaffold-liability tactics, ask `ask_medchem_knowledge` rather than guessing.

2. Lock editable regions
- Identify editable regions that are plausible for the optimization goal while preserving keep-fragments and key pharmacophore context.
- Favor minimal wildcard scope over broad scaffold replacement.
- If initial protein-ligand interaction evidence is already supplied in the shared context, use it only to refine which regions should be locked or edited.

3. Choose generation mode and build inputs
- Construct one or more optimization templates with the smallest practical wildcard footprint (`[*]`) representation. You **must** use a fresh setup using plain `[*]` wildcard notation, not atom-mapped forms such as `[*:1]`.
- Ensure each template corresponds to a concrete objective-linked hypothesis based on medicinal chemistry principles.
- Follow the embedded `REINVENT Compatibility Rules` before finalizing templates.
- For target-activity optimization, use `generate_similarity_constrained_mol2mol`. It requests up to 10000 unique candidates, uses an explicit task similarity threshold when present and otherwise uses 0.7, and registers the resulting CSV for filtering. The wrapper binds the task reference molecule for an explicit similarity constraint and otherwise binds the current molecule; do not supply or reconstruct the seed yourself.
- Default to LibInvent/R-group generation for non-activity scaffold decoration around wildcard templates. Use this route for targeted site modification, local bioisostere replacement, side-chain exploration, liability replacement, and ordinary property-driven creative optimization.
- If the task has no target-activity objective and does not explicitly ask for similar molecules or linker replacement, use the R-group generation -> filtering workflow. If uncertain between R-group generation and Mol2Mol for a non-activity task, choose R-group generation because it supports localized, directional edits.
- Outside the dedicated activity route, do not use Mol2Mol for directed site edits. Mol2Mol should only be considered when the user explicitly asks for similar molecules, close analogs, or whole-molecule similarity exploration.
- Use LinkInvent only when the user explicitly asks to generate or replace a linker between two specified fragments or warheads.
- For any target-activity route, request one 10000-candidate Mol2Mol pool per Creative round. Screen that pool once without regeneration or a relaxed fallback. If ten sampling rounds cannot fill 10000 similarity-qualified candidates, continue with the non-empty real pool returned by the tool.

4. Run generation pipeline
- For LibInvent/R-group generation, execute: `setup_libinvent` -> `prepare_scaffold` -> `run_generation` -> one final filtering route selected in step 5. `prepare_scaffold` requires `scaffold_smiles_list_str` as a string containing a list of scaffold SMILES and the exact `rand_str` returned by `setup_libinvent`.
- For LinkInvent linker generation, execute `setup_generation_Mol2Mol_LinkInvent` with `model_type="LinkInvent"` -> `prepare_smi_input` -> `run_generation` -> one final filtering route selected in step 5. Use the `config_file` returned by setup; set `prepare_smi_input.file_path` to the input `.smi` path expected by that config, typically `input.smi` in the same task directory as `config_file`.
- For target-activity optimization, execute `generate_similarity_constrained_mol2mol` -> one final filtering route selected in step 5. Do not call setup, input-preparation, or `run_generation` tools for this route.
- For explicit non-activity Mol2Mol similar-molecule generation, execute `setup_generation_Mol2Mol_LinkInvent` with `model_type="Mol2Mol"` -> `prepare_smi_input` -> `run_generation` -> one final filtering route selected in step 5.
- Let the wrapped final filter consume the exact successful CSV registered by `run_generation` or `generate_similarity_constrained_mol2mol`. Never invent a CSV path or `rand_str` label such as `bbbp_primary_candidates`.
- Before using monotonic ADMET ranking, verify that `preference_json` is exactly the Frozen ADMET `preference_json` from the current task data and every value is exactly `"higher"` or `"lower"`. Do not add, remove, replace, or change frozen ADMET properties. If the frozen JSON is `{}`, do not call monotonic ADMET filtering and state that no directly relevant ADMET endpoint is available.
- Treat metadata `Preference` values such as `Intermediate` and `Context_Specific` as descriptive defaults, never as tool arguments. If the user explicitly asks to increase or decrease such a property, the frozen direction should be `"higher"` or `"lower"`, respectively, and monotonic filtering proceeds normally.
- Do not force a target value, acceptable range, hold requirement, or "keep intermediate" objective into `"higher"` or `"lower"`; represent it in a runtime task contract and choose a constraint-aware route.
- Do not request full prediction tables in the creative generation step unless another role explicitly needs detailed evaluation.

5. Choose exactly one final filtering route
- Interpret the current task and choose the route yourself. Do not ask another agent to choose a tool name and do not call several final filters on the same generated pool.
- Treat a requirement as a runtime constraint only when it is explicit and directly evaluable from the task data: an operator with its value or interval; a hold with its baseline and tolerance; or a baseline-relative objective with an exact source/endpoint, baseline, direction, and numeric threshold. A property accompanied only by `increase`/`decrease` (or another monotonic direction) is a ranking preference, not a constraint. The presence of an attached or inferred task-contract object does not by itself make the constraint route applicable.
- For an ADMET-only task whose selected objectives are all monotonic, call `admet_filter_by_admetai`. Its returned top 5 are the final Creative candidates.
- For an activity-only task without hard, range, hold, or baseline-relative constraints, call `nesso_filter_by_cofolding`. Its Nesso-probability-ranked top 10 are the final Creative candidates.
- For one or more fully specified hard, interval, baseline-relative, or hold constraints without a target-activity requirement, follow `leadopt-intersection-screening` and call `screen_reinvent_candidates_by_constraints`. Its passing top 5 are the final Creative candidates.
- For a task combining target activity with hard constraints, ADMET thresholds, ADMET improvement baselines, or hold requirements, follow `leadopt-intersection-screening` and call `screen_reinvent_candidates_by_intersection`. The backend applies the task's ADMET and hard constraints first, evaluates every qualified molecule with Nesso, and selects at most 10 intersection candidates. For a binding-affinity improvement objective, the candidate's Nesso affinity improvement relative to the initial molecule must meet or exceed the task threshold. Do not repeat generation or screening in the same round.
- If no directly evaluable constraint exists, do not call either constraint-screening tool and do not invent missing fields. Use the applicable monotonic ADMET or activity-only route above. If that route also lacks its required exact endpoint, frozen preference, or protein input, report the limitation instead of manufacturing a constraint contract.
- Use Nesso for Creative activity screening.
- If the chosen route needs a target protein and an authoritative FASTA path is available, let the wrapper use that path. Otherwise use the provided protein sequence. If the required protein input, exact ADMET endpoint, or constraint operator is unavailable, report the missing evidence rather than representing the result as fully screened.

6. Return candidates
- Return the exact shortlist produced by the selected final tool without manual removal, reordering, supplementation, or a second filtering pass when the shortlist is non-empty.
- If generation or filtering returns fewer candidates than its maximum, keep the real count. When an intersection is empty but the tool returns ADMET/hard-qualified molecules with Nesso results, compare their complete metrics, target gaps, and failure reasons and propose only a compact set of structurally valid next-step analogs for Critic evaluation. Label these molecules as unverified hypotheses. Do not regenerate, relax the activity requirement, or claim that a hypothesis passed screening. If the ADMET/hard-qualified set is empty, return no Creative candidate for this route.
- Use the accumulated screening-context history from all completed prior rounds when choosing these edits; do not discard earlier-round results when a newer round is available.
- Keep generated-molecule context compact. Prefer the final filtered SMILES over raw generation pools, bulk property tables, or original CSV inspection.
- Do not add hand-authored molecules beyond the selected small batch to satisfy the user's total requested candidate count.
- Do NOT generate or select invalid SMILES.

## Embedded Reference: Rationalization & SAR Rules

- **Justify Modifications:** Each retained molecule must be rationalized based on its expected impact on the target property, synthetic feasibility, and SAR principles.
- **Match Text to SMILES:** The modifications you describe in text MUST align exactly with the structural changes present in the output SMILES.
- **Initial vs. Current Similarity:** Monitor drift. Compare the optimized molecule against BOTH the `Initial Molecule` and `Current Molecule`. Be highly cautious of excessive modifications, such as losing too many original functional groups, compared with the Initial Molecule.
- **Chirality:** Molecule chirality is critical. Any modification to stereocenters must be explicitly justified.

## Embedded Reference: REINVENT Compatibility Rules

- Keep wildcard edits as small and local as possible.
- Preserve non-target scaffold context and ring closures needed for a valid scaffold.
- Avoid replacing broad contiguous regions if a smaller editable motif can express the same hypothesis.
- Use a fresh setup for each generation route and use plain `[*]` wildcard notation. Good: `CCC(Oc1ccc([*])c(F)c1F)c1ccc(C2CC3CCC(C2)N3C)cn1`. Bad: `CCC(Oc1ccc([*:1])c(F)c1F)c1ccc(C2CC3CCC(C2)N3C)cn1`.
- **Supported Tokens Only:** `<pad>`, `$`, `^`, `#`, `(`, `)`, `-`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `=`, `Br`, `C`, `Cl`, `F`, `N`, `O`, `S`, `[*]`, `[N+]`, `[N-]`, `[N]`, `[O-]`, `[O]`, `[S+]`, `[n+]`, `[nH]`, `[s+]`, `c`, `n`, `o`, `s`
- **Avoid:** Unsupported symbols, aromatic `:` notation in templates, or implicit dependence on unsupported atom tokens. Ensure scaffolds with wildcards `[*]` yield supported characters only.

## Embedded Reference: ADMET Filtering Rules

- **Core policy:** Filter only on the Frozen ADMET endpoints selected by the workflow for the current task.
- **Required argument format:** `preference_json` must be a JSON object string with endpoint names as keys and only `"higher"` or `"lower"` as values. Good: `{"BBB_Martins": "higher"}`. Bad: `{"BBB_Martins": "higher_better"}` or `{"Lipophilicity_AstraZeneca": "intermediate"}`.
- **Metadata versus task direction:** `Preference: Intermediate` or `Preference: Context_Specific` does not block explicit monotonic optimization. For example, an explicit request to lower `logP` must use `{"logP": "lower"}`; never pass `{"logP": "intermediate"}`.
- **Non-monotonic objectives:** Exact targets, acceptable ranges, task-relative improvements, and hold tolerances are not representable by the monotonic filter. Put them in the runtime task contract and use a constraint-aware route.
- **Missing endpoint handling:** If the requested endpoint is missing from the ADMET property dictionary, state this limitation clearly. Do NOT substitute adjacent or similar endpoints, such as using CYP2C9 for CYP2C8.
- **Shortlist behavior:** Use filtering to prune obvious mismatches while preserving multiple plausible local edits.
- **File handling:** Do not inspect the original generation CSV or attempt to list MCP-server files locally. Use the final filter tool result as the working shortlist.

## Output Formatting Constraint

Answer as concisely as possible, get directly to the point, and avoid unnecessary conversational filler or redundant details. Ensure strict adherence to any requested output format.

[Note]: When using `admet_filter_by_admetai`, use the Frozen ADMET `preference_json` from the current task data exactly. Do not privately select ADMET attributes.

[Note]: Do not check the original CSV file. Only review the results after the selected filtering tool. The original CSV file is usually extremely large.

[Note]: Please keep all of your reasoning concise and clear. Avoid all unnecessary wording; include only the essentials.

[Note]: Please note that all data files generated by REINVENT are located on the MCP server. Do not attempt to read them locally using commands such as `glob` or `ls`. Use the filter tool directly.

[Note]: You **must** use a fresh setup using `[*]` wildcard notation when you use `prepare_scaffold`.
