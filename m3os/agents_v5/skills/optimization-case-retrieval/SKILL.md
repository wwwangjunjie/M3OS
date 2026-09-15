---
name: optimization-case-retrieval
description: Rational-only workflow for retrieving scaffold-similar molecular evolution subgraphs, extracting reusable transformation signals, and feeding case evidence into medicinal chemistry strategy synthesis without mechanically copying examples.
---

# optimization-case-retrieval

## Overview

Use this skill only in rational medicinal design. The case tool now retrieves molecular evolution subgraphs from property-specific DAGs, not isolated functional-group-matched molecule pairs. Cases are an evidence stream for strategy synthesis; they do not replace medicinal chemistry knowledge, SAR judgment, pharmacophore reasoning, or task constraints.

## Instructions

1. Extract the current scaffold
- First call `extract_molecule_scaffold` with the current molecule SMILES.
- Use the returned `scaffold_smiles` as the structural context for case retrieval. Do not use `generic_scaffold_smiles` for this workflow.
- If `scaffold_status` is not successful or `scaffold_smiles` is missing, state that scaffold-based case retrieval is not available for this molecule and rely on medicinal chemistry knowledge plus task constraints.

2. Query scaffold-similar evolution subgraphs
- Then call `query_evolutionary_optimization_cases` with the current molecule SMILES and the target ADMET property or properties.
- The case tool independently extracts `scaffold_smiles` again for retrieval, so check that its reported query scaffold matches the scaffold you inspected from `extract_molecule_scaffold`.
- The case tool checks an in-process cache before searching. If the same `scaffold_smiles`, property, and subgraph limit were queried earlier in this session, reuse the returned cached result as current context instead of asking for a fresh search.
- Request a compact number of examples. The tool caps results at 3 evolution subgraphs per property.

3. Read the returned subgraphs as graph evidence
- Treat each returned subgraph as a historical local evolution map: every listed edge is an optimization case from source molecule to target molecule.
- Use the edge fields directly: source/target SMILES, IUPAC names, Modification, Reason, `diff_before_opt`, `diff_after_opt`, template, and `evidence_count`.
- When a large subgraph is truncated, respect the displayed/omitted counts. Do not claim the displayed edges are the complete graph.

4. Extract reusable transformation signals
- Identify repeated or well-supported transformation motifs near the current scaffold context.
- Summarize what property problem each transformation was intended to solve and the structural scope of the edit.
- Note applicability limits: scaffold mismatch, excessive edit amplitude, conflicts with keep-fragments, pharmacophore risk, or reasons the historical edit should not be copied directly.

5. Integrate before designing
- Combine case-derived signals with retrieved medicinal chemistry knowledge and the current pharmacophore/SAR hypothesis before proposing molecules.
- Use cases to justify edit directions, not to mechanically copy source-to-target transformations.
- Do not stop after retrieval. Pass the transformation signal, applicability limits, and confidence into the case-guided design step.
- If no relevant case signal is found, state that briefly and rely on medicinal chemistry knowledge plus task constraints.

[Note]: Keep the case summary concise and decision-oriented. Include only evidence that changes the design strategy.
