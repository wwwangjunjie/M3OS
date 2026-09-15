---
name: case-guided-medicinal-design
description: Rational Designer workflow for converting shared analysis, medicinal chemistry knowledge, retrieved optimization cases, and pharmacophore/SAR judgment into a compact set of goal-calibrated, explainable molecular edits. Use when final molecules should be grounded in evidence and the edit amplitude should follow the task objective, from local tuning to supported motif replacement, without defaulting to tiny changes.
---

# case-guided-medicinal-design

## Overview

This skill defines the workflow for evidence-backed medicinal chemistry design. Choose the edit amplitude from the goal, shared diagnosis, medicinal chemistry knowledge, retrieved cases, and pharmacophore/SAR context. Do not treat high similarity or the smallest possible atom-level edit as a default objective; local tuning, moderate motif swaps, and stronger liability-focused replacements are all valid when justified.

## Instructions

1. Build the pharmacophore/SAR baseline
- Infer the likely pharmacophore from the current and initial molecules before proposing edits: hydrogen-bond donors/acceptors, aromatic or hydrophobic features, ionizable centers, key vectors, stereochemical presentation, and any interaction hypotheses implied by context.
- Treat functional-group or fragment listings as partial structural evidence only; they are not a substitute for pharmacophore analysis.
- Protect likely pharmacophore elements and SAR continuity unless the objective, risk alerts, medicinal chemistry knowledge, or cases justify changing them.
- Do not default to deleting substituents or shrinking the molecule. Deletion is acceptable only when the removed portion is a clear liability and the rationale explains why pharmacophore integrity, scaffold continuity, and the optimization goal remain protected.

2. Ground in evidence
- Treat shared current-analysis context as the default diagnosis baseline. Do not repeat full molecule diagnosis when reliable shared analysis is already provided; add only incremental findings that materially affect the design decision.
- Check role memory, lightweight medicinal chemistry answer memory, Project Manager Brief, and uploaded-document/table cues before requesting new evidence.
- Start from shared analysis, retrieved medicinal chemistry knowledge, and relevant optimization cases.
- When accumulated screening context is available, compare all prior-round ADMET/hard-qualified molecules, Nesso activity outcomes, target gaps, and failure reasons. Use those measured contrasts to choose edits, while treating any new molecule as unverified until Critic evaluation.
- Use retrieved cases to identify transformation motifs that are chemically plausible for the current molecule.
- Ignore case patterns that require broader scaffold shifts than the current task allows, but keep moderate motif swaps when knowledge and cases support them.
- If medicinal chemistry guidance is missing for the exact scaffold, property, risk, uploaded-document/table cue, or edit-amplitude decision, ask for focused guidance through the medicinal chemistry knowledge interface before finalizing strategies.

3. Turn evidence into strategies
- Keep reasoning and tool usage tied to properties that directly support the current optimization goal.
- Use the workflow-selected frozen ADMET endpoints and directions exactly. Do not privately add, remove, replace, or substitute ADMET endpoints. If a requested property is unavailable from tools, state that limitation clearly and fall back to medicinal chemistry judgment.
- Translate the best-supported patterns into a small set of concrete edit strategies.
- Favor interpretable, goal-calibrated edits that preserve user-required fragments and the likely pharmacophore.
- Include a range of local and moderate strategies when chemically defensible; do not collapse the batch into near-identical micro-edits.
- Prefer bioisosteric replacement, polarity tuning, steric/electronic modulation, local substituent changes, conformational control, or solubilizing/transport-aware appendage edits before removal.
- Follow the embedded `Evidence-Guided Design Decision Rules` when deciding between multiple plausible edit directions.

4. Construct candidates directly
- Generate a limited number of candidates by applying the selected strategies directly to the current molecule.
- Keep candidates synthetically plausible, valence-correct, and consistent with the intended mechanism of improvement.

5. Keep rationale strict
- For every candidate, explain the intended property impact using either medicinal chemistry theory or the retrieved case evidence.
- Ensure the stated rationale matches the exact atoms or motifs that were changed.
- Include a strategy synthesis before final molecules: key medicinal chemistry guidance, case-derived transformation signal, and how they were combined into the chosen edit amplitude.

6. Keep scope controlled
- Prefer a set of high-confidence candidates over a large speculative list.
- Do not add molecules solely to satisfy a user-requested total candidate count. Return the smaller defensible subset when necessary.

## Embedded Reference: Evidence-Guided Design Decision Rules

Use this reference when several case-guided edit directions are available.

### Prioritization order
1. Preserve user-required fragments and shared keep-fragments.
2. Preserve known or strongly inferred pharmacophore motifs.
3. Choose edits with a clear mechanism tied to the target property.

### Avoid
- Large multi-site edits unless clearly required by the task and justified by evidence.
- Repeating only methyl/halogen micro-edits when the optimization objective requires a stronger property shift.
- Rationales that cite a case pattern but implement a materially different transformation.

[Note]: Please keep all of your reasoning concise and clear. Avoid all unnecessary wording; include only the essentials!!!
