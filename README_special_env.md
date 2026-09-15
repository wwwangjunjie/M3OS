# `accfg` compatibility note

The validated M3OS workflow uses `accfg>=0.0.8` with two local compatibility
changes. This patch is not applied automatically and is not redistributed as a
modified third-party package.

## Required changes

1. Replace `ProcessPoolExecutor` with `ThreadPoolExecutor` in the installed
   `accfg` package. This avoids process serialization and worker-startup issues
   in the M3OS execution environment.
2. In `accfg`'s `compare.py`, update `process_unique_fgs_atoms()` so that it does
   not raise `ValueError` when `len(unique_atom_list) != number`. Preserve the
   filtered `unique_atom_list` and append it to the result.

The relevant logic should behave as follows:

```python
def process_unique_fgs_atoms(unique_fgs, mapped_atoms):
    unique_fgs_atoms = []
    for fg_name, number, atom_list in unique_fgs:
        unique_atom_list = []
        if number == len(atom_list):
            unique_fgs_atoms.append((fg_name, number, atom_list))
            continue
        for atom_set in atom_list:
            if set(atom_set).issubset(set(mapped_atoms)):
                continue
            unique_atom_list.append(atom_set)
        unique_fgs_atoms.append((fg_name, number, unique_atom_list))
    return unique_fgs_atoms
```

## Reproducible practice

Apply the changes in a project-local environment, record the exact `accfg`
version and patch diff, and do not modify a shared system installation. After
upgrading `accfg`, re-check whether the workaround is still necessary.

Locate and inspect the installed package with:

```bash
uv run python -c 'import pathlib, accfg; print(pathlib.Path(accfg.__file__).parent)'
grep -RInE 'ProcessPoolExecutor|ThreadPoolExecutor' .venv/lib/python*/site-packages/accfg
```

Run the M3OS import smoke test after applying the patch:

```bash
uv run --frozen python -c 'from m3os.agents_v5 import M3OSChatSession; print("M3OS import OK")'
```
