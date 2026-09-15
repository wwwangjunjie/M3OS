"""System prompt loader utilities for agent roles."""

from pathlib import Path
from string import Template


def _template_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "system" / filename


def _load_template(filename: str) -> Template:
    path = _template_path(filename)
    return Template(path.read_text(encoding="utf-8"))


def _build_additional_context_block(additional_context: str) -> str:
    if not additional_context:
        return ""
    return f"[USER PROVIDED ADDITIONAL CONTEXT]\n{additional_context}\n"


def render_main_system_prompt() -> str:
    template = _load_template("main_system.md")
    return template.safe_substitute()


def render_auditor_system_prompt() -> str:
    template = _load_template("auditor_system.md")
    return template.safe_substitute()


def render_rational_system_prompt(
    additional_context: str,
) -> str:
    template = _load_template("rational_system_v3.md")
    return template.safe_substitute(
        additional_context_block=_build_additional_context_block(additional_context),
    )


def render_creative_system_prompt(
    property_meta_info: str,
    additional_context: str,
) -> str:
    template = _load_template("creative_system_v3.md")
    return template.safe_substitute(
        property_meta_info=property_meta_info or "{}",
        additional_context_block=_build_additional_context_block(additional_context),
    )


def render_critic_system_prompt(
    property_meta_info: str,
    additional_context: str,
) -> str:
    template = _load_template("critic_system_v3.md")
    return template.safe_substitute(
        property_meta_info=property_meta_info or "{}",
        additional_context_block=_build_additional_context_block(additional_context),
    )


def render_report_system_prompt(
    additional_context: str,
) -> str:
    template = _load_template("report_system.md")
    return template.safe_substitute(
        additional_context_block=_build_additional_context_block(additional_context),
    )


def render_medchem_retrieval_system_prompt(
    additional_context: str,
) -> str:
    template = _load_template("medchem_retrieval_system.md")
    return template.safe_substitute(
        additional_context_block=_build_additional_context_block(additional_context),
    )
