---
name: candidate-quality-gating
description: Use this skill to evaluate all current-round candidates with goal-aligned endpoint selection, medicinal-chemistry risk review, and comparative ranking against the parent molecule.
---

# candidate-quality-gating

## Overview

This skill defines the end-to-end workflow for quantitative plus qualitative candidate evaluation.

## Instructions

1. Use frozen ADMET endpoints
- Use the Frozen ADMET preference_json selected by the workflow in the current task data.
- Do not add, remove, replace, or change ADMET endpoints or directions.
- Treat metadata `Preference` values such as `Intermediate` and `Context_Specific` as descriptive defaults, not tool arguments. An explicit user request to increase or decrease that property should already be frozen as `"higher"` or `"lower"`, respectively; use that frozen direction exactly.
- Do not encode a target value, acceptable range, or "keep intermediate" objective as an arbitrary monotonic direction. `admet_predict_by_admetai` can return raw selected values for comparison, but `preference_json` cannot represent distance to a target or range membership. State this limitation when it affects evaluation.
- If the frozen JSON is `{}`, state that no directly relevant ADMET endpoint is available before final ranking.
- If a requested property is unavailable from tools, state that limitation explicitly and fall back to medicinal chemistry judgment rather than substituting an unrelated endpoint.

2. Run quantitative scoring
- Focus reasoning and tool usage on properties that are directly tied to the current optimization goal.
- Submit the complete current-round candidate list to each applicable prediction tool. Do not classify candidates as Creative or Rational for prediction purposes and do not omit candidates believed to have been scored earlier; the tool backend reuses persisted raw predictions and calculates only cache misses.
- For a task with a lead-optimization contract, call `evaluate_candidate_constraints` on the complete Creative-plus-Rational list so deterministic hard constraints, ADMET objectives, and hold tolerances are checked with one task definition. Do not also call `admet_predict_by_admetai` for the same contract.
- For an active monotonic ADMET objective without a lead-optimization contract, call `admet_predict_by_admetai` using the Frozen ADMET preference_json exactly.
- For an active target-activity objective, follow `activity-consensus-evaluation` and call `evaluate_nesso_activity` once for the complete Creative-plus-Rational list. Use its Nesso binding probability, Nesso affinity value, task-pass tier, and deterministic rank.
- Match evidence by the SMILES returned in the prediction records. Do not copy numerical metrics from generator prose, reconstruct them from memory, or ask another agent to relay them.
- Compare candidate values against the Current Molecule rather than reading them in isolation.
- If an activity objective lacks both an authoritative FASTA path and a complete target sequence, state that activity evaluation is unavailable.

3. Perform medchem risk review
- Check toxicophore liability, instability risk, synthetic feasibility, and pharmacophore integrity.
- Penalize candidates that improve a target metric at the cost of obvious medicinal chemistry liabilities.

4. Rank comparatively
- Preserve all candidates in the output.
- Rank them by overall objective fit relative to the parent molecule and project constraints.
- Provide concise evidence-based rationale for each candidate.

5. Keep reward support clear
- Make it easy for downstream workflow steps to understand why a candidate scored well or poorly.
- Follow the embedded `Evaluation Rubric` when a consistent ranking rubric is needed.

## Embedded Reference: Evaluation Rubric

Use this rubric when comparing current-round candidates against the parent molecule.

### Required dimensions
- Goal fit: does the candidate move the target endpoint or property in the correct direction?
- Parent continuity: does it preserve essential scaffold and pharmacophore context?
- Medchem risk: does it introduce toxicophore, instability, or synthesis concerns?
- Evidence quality: are the claims supported by direct tool outputs, case evidence, or established medicinal chemistry reasoning?

### Ranking behavior
- A candidate with modest numerical improvement but clean medchem profile can outrank a numerically stronger but risky analog.
- If endpoint support is missing, say so explicitly and rank mainly on medicinal chemistry judgment.
- Keep all candidates in the final output, even low-ranked ones.

### Common penalties
- Large unjustified edit distance from the parent molecule.
- Structural alerts or obvious instability motifs.
- Loss of likely key binding or recognition features.
- Claims that are not supported by the actual structural modification.

[Note]: When using `admet_predict_by_admetai`, use the Frozen ADMET preference_json from the current task data exactly. Do not privately select ADMET attributes.

[Note]: Please keep all of your reasoning concise and clear. Avoid all unnecessary wording; include only the essentials!!!
