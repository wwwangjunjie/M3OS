---
name: protein-context-check
description: Interpret protein-ligand interaction evidence and translate contacts into medicinal-chemistry design hypotheses.
---

# protein-context-check

## Overview

This skill turns protein-ligand interaction records into concise, uncertainty-aware medicinal-chemistry evidence.

## Instructions

1. Read the interaction result
- Read every interaction entry as `[count, details]`; the count is the number of detected contacts and the second item contains the corresponding records.
- Distinguish hydrogen bonds, hydrophobic contacts, water bridges, salt bridges, pi stackings, pi-cation contacts, halogen bonds, and metal complexes.
- A zero count means that no contact of that type met the extractor's geometric rules; it does not establish absence of binding or activity.

2. Convert contacts into design evidence
- Summarize key residues, contact types, ligand features to preserve, and plausible exposed vectors.
- Separate directly observed contacts from inferred pharmacophore or SAR hypotheses.
- Connect proposed modifications to the evidence without claiming that unmodeled molecules retain the same pose or contacts.

3. Preserve uncertainty
- Treat structure prediction and interaction extraction as supporting evidence, not the sole decision maker.
- Do not overstate confidence from one predicted pose.
- If the interaction record is empty or incomplete, state that limitation and continue with the available textual and medicinal-chemistry evidence.
