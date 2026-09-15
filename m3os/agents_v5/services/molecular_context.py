"""Helpers for controlling explicit molecular auxiliary context in prompts."""

from __future__ import annotations


_AUXILIARY_FIELD_MARKERS = (
    "iupac",
    "functional group",
    "fragment information",
)


def without_explicit_molecular_auxiliary_fields(prompt: str) -> str:
    """Remove prompt lines that explicitly expose IUPAC or fragment fields.

    The SMILES-only benchmark ablation still lets models reason from SMILES,
    but it must not hand them pre-computed names, functional groups, or labels
    implying that those auxiliary values are available.
    """
    lines = []
    for line in str(prompt or "").splitlines():
        lowered = line.lower()
        if any(marker in lowered for marker in _AUXILIARY_FIELD_MARKERS):
            continue
        lines.append(line)
    return "\n".join(lines).strip()
