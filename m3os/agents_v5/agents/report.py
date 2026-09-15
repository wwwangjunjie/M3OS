"""Report agent for final molecular optimization SAR reports."""

import json
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from m3os.agents_v5.agents.base import BaseAgent
from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.core.models import ReportBlueprint
from m3os.agents_v5.core.retry_utils import invoke_with_retry
from m3os.agents_v5.prompts.system_loader import render_report_system_prompt
from m3os.agents_v5.services.mcp_client import MCPClientManager


REPORT_BLUEPRINT_PROMPT = """Plan an agentic SAR HTML report blueprint from the completed MCGS molecular optimization graph.

Hard constraints:
- Write every narrative field in the requested report_language: {report_language}.
- Use only the exact candidate SMILES supplied below. Do not invent candidates, canonicalize, shorten, or rewrite SMILES.
- complete_candidate_smiles must contain every candidate SMILES from candidate_nodes_json exactly once.
- molecule_cards target count is max_detailed_cards. If candidate_count <= 30, every candidate must have a detailed card. If candidate_count > 30, include at least max_detailed_cards cards and cover every strategy group represented in the graph.
- If a node is marked unreachable, keep it in the complete candidate audit but do not make it high priority or first-round synthesis.
- Medicinal chemistry judgments must be tied to supplied action, generator rationale, critic rationale, selected_reason, ADMET policy, node properties, or RDKit properties. If evidence is indirect, phrase it as expected direction, risk, or recommended validation.
- Do not invent precise experimental values, assay results, literature citations, ADMET predictions, or synthesis route details that are not in the inputs.
- Use asset_id values exactly as supplied. Candidate images are referenced in HTML as {{{{image:M001}}}} and root-vs-candidate difference images as {{{{image_diff:M001}}}}.

[Original task]
{user_prompt}

[Project manager brief]
{project_manager_brief}

[Optimization goal]
{optimization_goal}

[Root compound]
SMILES: {initial_smiles}
IUPAC: {initial_iupac}
Fragments: {initial_fragments}

[Report controls]
report_language: {report_language}
coverage_mode: {coverage_mode}
top_n_focus: {top_n_focus}
first_round_n: {first_round_n}
max_detailed_cards: {max_detailed_cards}
candidate_count: {candidate_count}
style_reference: {style_reference}
previous_validation_error: {report_validation_error}

[ADMET/optimization policy]
{admet_policy_json}

[Final search statistics]
{graph_statistics}

[Candidate molecule nodes, sorted by reachability, score, and UCT]
{candidate_nodes_json}

[Report asset summary; refer to assets by asset_id in HTML using 2D topology placeholders {{{{image:M001}}}} and root-vs-candidate difference placeholders {{{{image_diff:M001}}}}]
{report_assets_json}

[Current search path and selection information]
{current_info_list_json}

[Required report outline]
1. Cover/executive recap
2. Data source and limitations
3. Root molecule
4. MCGS graph overview
5. Strategy overview
6. Detailed molecule cards by strategy group
7. Complete candidate audit table
8. First-round synthesis suggestions
9. Risks and expert questions

[Note]: Keep reasoning concise and evidence-bound. Return only the ReportBlueprint object.
"""


REPORT_HTML_PROMPT = """Generate a complete, self-contained HTML medicinal chemistry optimization report from the ReportBlueprint and assets.

Hard constraints:
- Return HTML only: no Markdown fence or commentary.
- Include <!doctype html>, <html lang="{report_language}">, <head>, <meta charset="utf-8">, <title>, inline <style>, and <body>.
- Do not use external resources, scripts, iframes, forms, remote images, or remote CSS.
- In-page navigation with <a href="#..."> is allowed.
- Use asset placeholders: {{{{image:parent}}}} for the root, {{{{image:M001}}}} for a candidate topology, and {{{{image_diff:M001}}}} for a root-vs-candidate difference. Do not emit raw base64.
- Prefer difference images in candidate cards or structural-change sections; green marks the common scaffold and orange marks changes relative to the root.
- Include every candidate SMILES in at least one overview table or card without rewriting it.
- Do not invent experimental measurements, literature citations, new candidates, or unavailable ADMET predictions. Describe experimental claims as recommended validation, expected direction, or risk.
- Bind medicinal-chemistry judgments to supplied Blueprint, action, rationale, properties, ADMET policy, or RDKit properties.
- Use a dark, compact, professional medicinal-chemistry design-review layout rather than a marketing page.
- Follow this outline: cover summary; data sources and limitations; root molecule; MCGS graph overview; strategy overview; detailed candidate cards by strategy group; complete candidate audit; first-round synthesis suggestions; risks and expert questions.
- Narrative language must match report_language: {report_language}.

[ReportBlueprint JSON]
{blueprint_json}

[Original task and goal]
{task_context_json}

[Report assets JSON]
{report_assets_json}

[Note]: Keep the HTML polished, compact, and professionally readable for medicinal chemists.
"""


class ReportAgent(BaseAgent):
    """Agent that creates structured final report content."""

    def __init__(
        self,
        config: AgentConfig,
        mcp_client: MCPClientManager,
        additional_context: Optional[str] = None,
    ):
        super().__init__(config, mcp_client, additional_context=additional_context)
        self.medchem_knowledge = ""

    @property
    def system_prompt(self) -> str:
        additional_context = self.additional_context
        if self.medchem_knowledge:
            knowledge_block = (
                "[RETRIEVED MEDICINAL CHEMISTRY KNOWLEDGE]\n"
                f"{self.medchem_knowledge}"
            )
            additional_context = (
                f"{additional_context}\n\n{knowledge_block}"
                if additional_context
                else knowledge_block
            )
        return render_report_system_prompt(
            additional_context=additional_context,
        )

    @property
    def output_schema(self) -> type:
        return ReportBlueprint

    @property
    def agent_role(self) -> str:
        return "report"

    async def get_tools(self) -> List[Any]:
        return []

    def update_knowledge(self, knowledge: str) -> None:
        """Update retrieved medicinal chemistry knowledge used by the report agent."""
        self.medchem_knowledge = str(knowledge or "").strip()
        self._agent = None

    def format_user_prompt(self, **kwargs: Any) -> str:
        """Backward-compatible alias for the blueprint planning prompt."""
        return self.format_blueprint_prompt(**kwargs)

    def format_blueprint_prompt(self, **kwargs: Any) -> str:
        """Format the report-planning request with stable JSON serialization."""
        payload = dict(kwargs)
        payload.setdefault("project_manager_brief", "")
        payload.setdefault("coverage_mode", "all")
        payload.setdefault("top_n_focus", 8)
        payload.setdefault("first_round_n", 5)
        payload.setdefault("max_detailed_cards", None)
        payload.setdefault("candidate_count", len(payload.get("candidate_nodes") or []))
        payload.setdefault("report_language", "zh")
        payload.setdefault("report_validation_error", "")
        payload.setdefault("style_reference", "dark_medicinal_design_board")
        payload["candidate_nodes_json"] = json.dumps(
            payload.get("candidate_nodes", []),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        payload["report_assets_json"] = json.dumps(
            payload.get("report_assets", {}),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        payload["current_info_list_json"] = json.dumps(
            payload.get("current_info_list", []),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        payload["admet_policy_json"] = json.dumps(
            payload.get("admet_policy", {}),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        payload["graph_statistics"] = json.dumps(
            payload.get("graph_statistics", {}),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        return REPORT_BLUEPRINT_PROMPT.format(**payload)

    async def plan_report(self, **kwargs: Any) -> ReportBlueprint:
        """Plan the final report as a structured blueprint."""
        query = self.format_blueprint_prompt(**kwargs)
        result = await self.invoke(query)
        structured = result.get("structured_response") if isinstance(result, dict) else result
        if isinstance(structured, ReportBlueprint):
            return structured
        return ReportBlueprint.model_validate(structured)

    async def write_html_report(
        self,
        *,
        blueprint: ReportBlueprint,
        report_assets: Any,
        task_context: Any,
    ) -> str:
        """Write complete HTML from a report blueprint and code-generated assets."""
        model = self._create_model()
        report_language = getattr(blueprint, "report_language", "zh") or "zh"
        query = REPORT_HTML_PROMPT.format(
            report_language=report_language,
            blueprint_json=json.dumps(
                blueprint.model_dump(),
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            task_context_json=json.dumps(
                task_context,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            report_assets_json=json.dumps(
                report_assets,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
        )
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query),
        ]

        async def _invoke() -> Any:
            return await model.ainvoke(messages)

        raw_result = await invoke_with_retry(
            _invoke,
            max_retries=3,
            base_delay=1.0,
            validate_fn=lambda value: bool(self._extract_model_response_text(value).strip()),
            operation_name=f"{self.__class__.__name__} HTML report writing",
        )
        return _strip_html_fences(self._extract_model_response_text(raw_result))


def _strip_html_fences(text: str) -> str:
    """Remove common Markdown wrappers from model-produced HTML."""
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("html"):
            cleaned = cleaned[4:].strip()
    return cleaned


async def create_report_agent(
    config: AgentConfig,
    mcp_client: MCPClientManager,
    additional_context: Optional[str] = None,
) -> ReportAgent:
    """Factory function to create a report agent."""
    agent = ReportAgent(config, mcp_client, additional_context)
    await agent.build()
    return agent
