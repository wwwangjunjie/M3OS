"""Multimodal molecule visual context helpers.

These helpers keep image bytes out of persisted workflow state while allowing
selected LLM calls to receive RDKit molecule drawings as true image blocks.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw

from m3os.agents_v5.tools.molecule_images import (
    generate_molecule_pair_difference_image,
    generate_molecule_topology_image,
)


PNG_MIME_TYPE = "image/png"


@dataclass
class MoleculeVisualContext:
    """A generated molecule image plus model-ready content blocks."""

    visual_id: str
    image_type: str
    smiles: str
    content_blocks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def included_in_model_call(self) -> bool:
        return bool(self.content_blocks) and not self.error


def text_block(text: Any) -> Dict[str, str]:
    return {"type": "text", "text": str(text or "")}


def build_topology_visual_context(
    *,
    smiles: Any,
    visual_id: str,
    label: str,
    iupac: Any = None,
    fragments: Any = None,
    image_prefix: str = "molecule_topology_visual",
) -> MoleculeVisualContext:
    """Generate a topology image and return model-ready visual context."""
    smiles_text = str(smiles or "").strip()
    context = MoleculeVisualContext(
        visual_id=visual_id,
        image_type="topology",
        smiles=smiles_text,
    )
    if not smiles_text:
        context.error = "No SMILES provided for topology visual context."
        context.content_blocks = [_visual_failure_block(context, label)]
        context.metadata = {
            "visual_id": visual_id,
            "image_type": "topology",
            "smiles": smiles_text,
            "included_in_model_call": False,
            "visual_context_error": context.error,
        }
        return context

    try:
        result = generate_molecule_topology_image(
            smiles=smiles_text,
            image_prefix=_visual_image_prefix(image_prefix, visual_id),
        )
        image_path = str(result.get("image_path") or "")
        image_block = _image_block_from_path(image_path)
        canonical = result.get("canonical_smiles") or smiles_text
        pre_text = (
            f"[MOLECULE VISUAL CONTEXT]\n"
            f"Visual ID: {visual_id}\n"
            f"Image type: 2D topology\n"
            f"Label: {label}\n"
            f"SMILES for this image: {canonical}\n"
            f"IUPAC for this image: {iupac if iupac is not None else 'Unknown'}\n"
            f"Fragments for this image: {fragments if fragments is not None else 'Unknown'}\n"
            "Use this 2D topology together with the SMILES/IUPAC/fragments when reasoning about the molecule."
        )
        post_text = (
            f"[VISUAL BINDING CONFIRMATION]\n"
            f"{visual_id} == {label} == SMILES: {canonical}"
        )
        context.content_blocks = [text_block(pre_text), image_block, text_block(post_text)]
        context.metadata = {
            "visual_id": visual_id,
            "image_type": "topology",
            "image_path": image_path,
            "relative_image_path": result.get("relative_image_path"),
            "smiles": smiles_text,
            "canonical_smiles": canonical,
            "included_in_model_call": True,
        }
        return context
    except Exception as exc:
        context.error = f"{type(exc).__name__}: {exc}"
        context.content_blocks = [_visual_failure_block(context, label)]
        context.metadata = {
            "visual_id": visual_id,
            "image_type": "topology",
            "smiles": smiles_text,
            "included_in_model_call": False,
            "visual_context_error": context.error,
        }
        return context


def build_pair_difference_visual_context(
    *,
    parent_smiles: Any,
    candidate_smiles: Any,
    visual_id: str,
    candidate_source_agent_type: Any = None,
    image_prefix: str = "molecule_pair_difference_visual",
) -> MoleculeVisualContext:
    """Generate a parent-candidate difference image with strict SMILES binding."""
    parent_text = str(parent_smiles or "").strip()
    candidate_text = str(candidate_smiles or "").strip()
    context = MoleculeVisualContext(
        visual_id=visual_id,
        image_type="pair_difference",
        smiles=candidate_text,
    )
    if not parent_text or not candidate_text:
        context.error = "Missing parent or candidate SMILES for pair difference visual context."
        context.content_blocks = [_visual_failure_block(context, "parent-candidate difference")]
        context.metadata = {
            "visual_id": visual_id,
            "image_type": "pair_difference",
            "parent_smiles": parent_text,
            "candidate_smiles": candidate_text,
            "candidate_source_agent_type": str(candidate_source_agent_type or "unknown"),
            "included_in_model_call": False,
            "visual_context_error": context.error,
        }
        return context

    try:
        result = generate_molecule_pair_difference_image(
            before_smiles=parent_text,
            after_smiles=candidate_text,
            image_prefix=_visual_image_prefix(image_prefix, visual_id),
        )
        image_path = str(result.get("image_path") or "")
        caption_text = _pair_caption_text(visual_id)
        _add_pair_caption_strip(image_path, caption_text)
        image_block = _image_block_from_path(image_path)
        source_text = str(candidate_source_agent_type or "unknown")
        before_canonical = result.get("before_canonical_smiles") or parent_text
        after_canonical = result.get("after_canonical_smiles") or candidate_text
        pre_text = (
            f"[MOLECULE VISUAL CONTEXT]\n"
            f"Visual ID: {visual_id}\n"
            f"Image type: parent-candidate difference\n"
            f"Parent/current SMILES: {before_canonical}\n"
            f"Candidate SMILES: {after_canonical}\n"
            f"Candidate source agent_type: {source_text}\n"
            "Image binding: the image immediately following this text belongs only to this Visual ID and Candidate SMILES.\n"
            "Color guide: green/shared MCS, orange/changed region, purple/stereochemistry change."
        )
        post_text = (
            f"[VISUAL BINDING CONFIRMATION]\n"
            f"{visual_id} == Candidate SMILES: {after_canonical}\n"
            f"{visual_id} parent/current SMILES: {before_canonical}\n"
            "Do not transfer visual observations from this image to any other candidate."
        )
        context.content_blocks = [text_block(pre_text), image_block, text_block(post_text)]
        context.metadata = {
            "visual_id": visual_id,
            "image_type": "pair_difference",
            "image_path": image_path,
            "relative_image_path": result.get("relative_image_path"),
            "parent_smiles": parent_text,
            "candidate_smiles": candidate_text,
            "before_canonical_smiles": before_canonical,
            "after_canonical_smiles": after_canonical,
            "candidate_source_agent_type": source_text,
            "caption_text": caption_text,
            "mcs_atoms": result.get("mcs_atoms"),
            "before_changed_atoms": result.get("before_changed_atoms"),
            "after_changed_atoms": result.get("after_changed_atoms"),
            "before_stereo_changed_atoms": result.get("before_stereo_changed_atoms"),
            "after_stereo_changed_atoms": result.get("after_stereo_changed_atoms"),
            "included_in_model_call": True,
        }
        return context
    except Exception as exc:
        context.error = f"{type(exc).__name__}: {exc}"
        context.content_blocks = [_visual_failure_block(context, "parent-candidate difference")]
        context.metadata = {
            "visual_id": visual_id,
            "image_type": "pair_difference",
            "parent_smiles": parent_text,
            "candidate_smiles": candidate_text,
            "candidate_source_agent_type": str(candidate_source_agent_type or "unknown"),
            "included_in_model_call": False,
            "visual_context_error": context.error,
        }
        return context


def contexts_to_metadata(contexts: List[MoleculeVisualContext]) -> List[Dict[str, Any]]:
    return [dict(context.metadata) for context in contexts if context.metadata]


def _visual_failure_block(context: MoleculeVisualContext, label: str) -> Dict[str, str]:
    return text_block(
        "[MOLECULE VISUAL CONTEXT WARNING]\n"
        f"Visual ID: {context.visual_id}\n"
        f"Image type: {context.image_type}\n"
        f"Label: {label}\n"
        f"SMILES: {context.smiles or 'Unknown'}\n"
        f"Visual image unavailable: {context.error or 'unknown error'}\n"
        "Proceed using SMILES/IUPAC/fragments and explicitly note that the visual image is unavailable if relevant."
    )


def _image_block_from_path(image_path: str) -> Dict[str, str]:
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if path.suffix.lower() != ".png":
        raise ValueError(f"Only PNG visual context is supported: {path}")
    image_bytes = path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {"type": "image", "base64": encoded, "mime_type": PNG_MIME_TYPE}


def _pair_caption_text(visual_id: str) -> str:
    return f"{visual_id} | Left = current parent | Right = candidate"


def _visual_image_prefix(image_prefix: str, visual_id: str) -> str:
    safe_visual_id = str(visual_id or "visual").strip().lower()
    return f"{image_prefix}_{safe_visual_id}"


def _add_pair_caption_strip(image_path: str, caption_text: str) -> None:
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found for caption: {path}")

    with Image.open(path) as image:
        base = image.convert("RGB")
        strip_height = 44
        width, height = base.size
        output = Image.new("RGB", (width, height + strip_height), "white")
        output.paste(base, (0, strip_height))
        draw = ImageDraw.Draw(output)
        draw.rectangle((0, 0, width, strip_height), fill=(245, 247, 250))
        draw.text(
            (14, 8),
            caption_text,
            fill=(20, 25, 32),
        )
        output.save(path, format="PNG")
