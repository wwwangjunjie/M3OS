---
name: nesso-activity-scoring
description: Use Nesso to screen or evaluate molecules when target binding or activity is an optimization objective and a target protein sequence is available.
---

# Nesso activity scoring

## Core policy

- Use Nesso as the default target-activity model. It scores protein-sequence and ligand-SMILES pairs without generating protein-ligand structures.
- The complete task target `protein_sequence` is required. Always reuse it exactly; never abbreviate, truncate, or reconstruct it from an identifier. `protein_id` is optional metadata and must not substitute for the sequence.
- In task-gated intersection screening and Critic evaluation, apply `Nesso_affinity_probability_binary` as the task-defined probability floor and rank passing molecules by affinity improvement relative to the initial molecule, using probability only as a tie-breaker. Treat both values as model outputs, not experimental measurements.

## Creative Explorer

- Use only `nesso_filter_by_cofolding` for target-activity filtering.
- For an activity-only objective, pass the exact Reinvent `generated_csv_path` as `candidate_csv_path` and set `top_n=10`; this standalone tool retains its defined probability ordering.
- For a combined ADMET and activity objective, use `screen_reinvent_candidates_by_intersection`; its backend evaluates the complete ADMET/hard-qualified set with Nesso probability and affinity gates before selection.
- The returned candidates are the final current-round set. Do not manually reorder, replace, or supplement them.

## Critic

- Use `evaluate_nesso_activity` for target-activity evaluation; its backend calls `nesso_predict_by_cofolding` and reuses cached results.
- Supply every current-round candidate SMILES and, when a direct comparison is needed, the current parent SMILES with the exact target sequence. Do not omit Creative candidates or decide which molecules need recomputation; the tool returns persisted predictions and scores only unseen protein–SMILES pairs.
- Report both `Nesso_affinity_pred_value` and `Nesso_affinity_probability_binary` for successfully scored molecules.
- Treat the SMILES-keyed tool response as the quantitative source of truth rather than generator prose or agent-to-agent metric transfer.
- Keep activity conclusions separate from ADMET and medicinal-chemistry judgments.

## Missing inputs and uncertainty

- If activity optimization is requested without a target protein sequence, state that Nesso scoring cannot run; do not infer a sequence from a name, identifier, or structure path.
- Do not call Nesso merely because a sequence is present when target activity is not an optimization objective.
- Treat Nesso results as predictive evidence and preserve uncertainty in downstream decisions.
