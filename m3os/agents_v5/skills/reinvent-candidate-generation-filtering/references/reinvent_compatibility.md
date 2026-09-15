# REINVENT Compatibility Rules

Use this reference when constructing wildcard inputs for external REINVENT MCP generation.

## Template rules
- Keep wildcard edits as small and local as possible.
- Preserve non-target scaffold context and ring closures needed for a valid scaffold.
- Avoid replacing broad contiguous regions if a smaller editable motif can express the same hypothesis.
- Use a fresh setup for each generation route.
- Use plain `[*]` wildcard notation in templates. Do not use atom-mapped wildcard notation such as `[*:1]`.
- Pass `prepare_scaffold.scaffold_smiles_list_str` as a string containing a list of scaffold SMILES, and pass the exact `rand_str` returned by `setup_libinvent`.
- Use `setup_libinvent` -> `prepare_scaffold` -> `run_generation` for LibInvent scaffold decoration.
- Use `setup_generation_Mol2Mol_LinkInvent` with `model_type="LinkInvent"` -> `prepare_smi_input` -> `run_generation` for linker generation between two wildcard-marked fragments.
- Use `setup_generation_Mol2Mol_LinkInvent` with `model_type="Mol2Mol"` only for explicit similar-molecule, close-analog, or whole-molecule similarity requests.
- For target-activity optimization, use `generate_similarity_constrained_mol2mol` instead of the setup/input/run sequence. Its wrapper binds the seed and task similarity threshold and registers the generated CSV.

## Token compatibility
REINVENT only supports the following tokens:
- `<pad>`, `$`, `^`, `#`, `(`, `)`, `-`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `=`, `Br`, `C`, `Cl`, `F`, `N`, `O`, `S`, `[*]`, `[N+]`, `[N-]`, `[N]`, `[O-]`, `[O]`, `[S+]`, `[n+]`, `[nH]`, `[s+]`, `c`, `n`, `o`, `s`

## Avoid
- Unsupported symbols or tokenization patterns.
- Aromatic `:` notation in templates.
- Templates that implicitly depend on unsupported atom tokens.
- Templates whose wildcards destroy the intended parent scaffold interpretation.
- Broad scaffold hopping unless the user explicitly requests it and the task evidence supports it.
