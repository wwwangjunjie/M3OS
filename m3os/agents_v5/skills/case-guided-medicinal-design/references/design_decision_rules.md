# Evidence-Guided Design Decision Rules

Use this reference when several case-guided edit directions are available.

## Prioritization order
1. Preserve user-required fragments and shared keep-fragments.
2. Preserve known or strongly inferred pharmacophore motifs.
3. Choose edits with a clear mechanism tied to the target property.
4. Match edit amplitude to the property problem: local moves for fine tuning, moderate motif swaps for liabilities that local moves cannot credibly solve, and larger focused replacements only when the task evidence supports them.

## Typical good moves
- Bioisosteric replacements that preserve geometry and interaction intent.
- Local polarity or lipophilicity tuning at a non-core substituent.
- Small heteroatom, ring, or substituent changes supported by similar case evidence.
- Moderate side-chain, appendage, or liability-motif replacement supported by medicinal chemistry knowledge plus case evidence.

## Avoid
- Broad scaffold hopping.
- Large multi-site edits unless clearly required by the task and justified by evidence.
- Repeating only methyl/halogen micro-edits when the optimization objective requires a stronger property shift.
- Rationales that cite a case pattern but implement a materially different transformation.
