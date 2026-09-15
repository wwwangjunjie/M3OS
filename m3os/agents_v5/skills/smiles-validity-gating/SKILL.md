---
name: smiles-validity-gating
description: Use this skill before final optimization or critic summaries whenever a response contains molecule SMILES; validate all final SMILES in batch, repair invalid generated SMILES up to three times, and drop unrepaired invalid molecules.
---

# smiles-validity-gating

## Overview

This skill keeps final molecule lists chemically parseable before downstream audit extraction.

## Instructions

1. Validate final SMILES before the audit-facing summary
- Before producing any final molecule list, collect every SMILES you intend to include.
- Call `validate_smiles_batch` once with the complete list.
- Treat a value of `valid` as valid. Treat any other value as an error message that must be addressed.

2. Repair generated molecules when possible
- For rational or creative generation, inspect each invalid SMILES and the intended modification direction.
- Replace the invalid SMILES with one chemically valid SMILES that preserves the same modification intent.
- Re-call `validate_smiles_batch` on the complete revised list after each repair attempt.
- You have at most three repair attempts for an invalid molecule.

3. Drop unrepaired invalid molecules
- If a molecule remains invalid after three repair attempts, omit that molecule from the final list.
- Do not include invalid SMILES in explanatory text, tables, or audit summaries.
- Do not create extra molecules solely to replace dropped invalid molecules. Return the smaller valid subset when needed.

4. Critic behavior
- The critic should not invent new candidate molecules.
- If a critic evaluation contains an invalid copied SMILES, correct it only when it clearly matches a valid candidate from the input; otherwise drop that evaluation row.
- Preserve the original generator provenance when evaluating or correcting copied SMILES.

5. Keep final outputs consistent
- The final table or bullet list must contain only validated SMILES.
- Make the action/modification, rationale, score, and properties match the corrected SMILES.
- Keep the final molecule count within the role's current-round budget, even after repairs.
