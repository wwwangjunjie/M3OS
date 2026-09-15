"""Pydantic models for structured input/output and data validation."""

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ==================== Molecule Generation Output Schemas ====================

class MoleculeOptimization(BaseModel):
    """Describes the optimization result of a single molecule.
    
    Used as output from Creative Molecule Explorer and Rational Medicinal Designer.
    """
    smiles: str = Field(
        description="Optimized SMILES string",
        examples=[
            "CCC(Oc1ccc(C)c(F)c1F)c1ccc(C2CC2)cn1",
        ],
    )
    modification_type: str = Field(
        description="Type of molecular modification, e.g., Bioisostere, Rigidification, Polar Group Addition. "
                   "Note: The modifications described should align with the actual operations performed on the molecule's SMILES."
                   "Please keep it as concise as possible, no redundant content, only the core.",
        examples=[
            "Carboxylic acid → methylated pyrrolidine; cyclopropylmethoxy → methylpyrrolidine-methyl",
        ],
    )
    rationale: str = Field(
        description="Explanation for the modification, referencing specific theories. "
                   "If an example from query_evolutionary_optimization_cases was referenced, include that specific case with SMILES before and after optimization. "
                   "If no relevant case was found, explicitly state 'No related optimization example'."
                   "Please keep it as concise as possible, no redundant content, only the core.",
        examples=[
            "Removes carboxylic acid (-37 Ų PSA) and ionization. Removes ether linkage (-9 Ų PSA) by direct cyclopropyl attachment, reducing H-bond acceptors from 6 to 4. Expected PSA ~46 Ų. Aligns with morphine→codeine precedent (methyl addition increases BBB 10-fold). Related optimization example: Row 6 (CO₂H → amine, aggressive PSA reduction).",
        ],
    )
    confidence_score: float = Field(
        description="Confidence score for the modification, ranging from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
        examples=[0.85],
    )


class MoleculeOptimizations(BaseModel):
    """A list containing multiple molecule optimization results."""
    optimizations: List[MoleculeOptimization] = Field(description="List of optimized molecules")


# ==================== Critic Evaluation Output Schemas ====================

class CriticMoleculeEvaluation(BaseModel):
    """Represents a single molecule's evaluation by the Critic."""
    smiles: str = Field(
        description="The SMILES string of the evaluated molecule",
        examples=[
            "CCC(Oc1ccc(C)c(F)c1F)c1ccc(C2CC2)cn1",
        ],
    )
    action: str = Field(
        description="The chemical modification operation relative to the current molecule, e.g., 'Ester to amide conversion'. This is not an accept/reject decision. Please keep it as concise as possible, no redundant content, only the core.",
        examples=[
            "CO₂H → CH₃; cyclopropylmethoxy → methylpyrrolidine-methyl",
        ],
    )
    rationale: str = Field(
        description="Explanation of the modification including relevant molecular pair cases. Please keep it as concise as possible, no redundant content, only the core.",
        examples=[
            "Dual modification eliminates acid ionization and ether H-bond acceptor. BBB_Martins 0.969 (+0.252 vs parent). Directly removes two key BBB barriers (acid + excess PSA). Simpler than pyrrolidine analog but lacks basic amine benefit.",
        ],
    )
    score: float = Field(
        description="Score for the modification, range 0.0 to 1.0",
        ge=0.0,
        le=1.0,
        examples=[0.87],
    )
    properties: Dict[str, Any] = Field(
        description="Properties related to optimization in dictionary format",
        examples=[{"BBB_Martins": 0.984}],
    )
    agent_type: str = Field(
        description="The type of agent that generated this molecule: 'creative_molecule_explorer' or 'rational_medicinal_designer'",
        examples=["creative_molecule_explorer", "rational_medicinal_designer"],
    )

    @field_validator('properties', mode='before')
    @classmethod
    def parse_properties(cls, v):
        """Parse properties if it's a JSON string."""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # If parsing fails, return as-is and let Pydantic handle the error
                return v
        return v


class CriticEvaluations(BaseModel):
    """Contains multiple molecule evaluation results from the Critic."""
    optimizations: List[CriticMoleculeEvaluation] = Field(description="List of evaluated molecules")


# ==================== Report Generation Output Schemas ====================

class ReportModificationSite(BaseModel):
    """A recommended site for chemical modification in the final report."""

    site_id: str = Field(description="Short site identifier, e.g. Site 1")
    site_name: str = Field(description="Human-readable structural region or position")
    recommendation: str = Field(description="Why this site should be modified and what to explore")


class ReportStrategyGroup(BaseModel):
    """A strategy family used to organize designed molecules in the report."""

    group_id: str = Field(description="Short group identifier, e.g. A")
    group_name: str = Field(description="Strategy family name")
    design_purpose: str = Field(description="Main design purpose of this strategy family")
    sar_output: str = Field(description="What SAR information this strategy is expected to produce")


class ReportDesignedMolecule(BaseModel):
    """A designed molecule entry for the final SAR report."""

    design_id: str = Field(description="Report design identifier, e.g. A1")
    smiles: str = Field(description="Designed molecule SMILES")
    strategy_group: str = Field(description="Strategy group id or name this molecule belongs to")
    design_logic: str = Field(description="Medicinal chemistry design logic")
    sar_exploration_direction: str = Field(description="Expected SAR exploration direction")
    synthetic_feasibility: str = Field(description="Synthetic feasibility assessment")
    priority: Literal["H", "M", "L"] = Field(description="Priority: H, M, or L")


class ReportFirstRoundCandidate(BaseModel):
    """A first-round synthesis recommendation."""

    design_id: str = Field(description="Design id of the recommended molecule")
    reason: str = Field(description="Why this molecule should be synthesized first")


class OptimizationReport(BaseModel):
    """Structured semantic content for the final HTML optimization report."""

    report_title: str = Field(description="Chinese report title")
    target_summary: str = Field(description="Concise optimization target summary")
    original_compound_label: str = Field(description="Label for the original compound")
    core_scaffold: str = Field(description="Core scaffold and overall structure summary")
    key_structural_features: List[str] = Field(description="Key structural features of the original molecule")
    recommended_modification_sites: List[ReportModificationSite] = Field(description="Recommended modification sites")
    strategy_groups: List[ReportStrategyGroup] = Field(description="Modification strategy families")
    designed_molecules: List[ReportDesignedMolecule] = Field(description="Designed molecule report entries")
    first_round_synthesis: List[ReportFirstRoundCandidate] = Field(description="Recommended first-round synthesis list")
    risk_notes: List[str] = Field(description="Risk notes and experimental validation recommendations")


class ReportBlueprintSection(BaseModel):
    """A planned section for an agentic HTML SAR report."""

    section_id: str = Field(description="Stable short section id, suitable for HTML anchors")
    title: str = Field(description="Section title in the report language")
    purpose: str = Field(description="Why this section belongs in the report")
    key_points: List[str] = Field(default_factory=list, description="Concise bullets to cover in the section")
    referenced_design_ids: List[str] = Field(
        default_factory=list,
        description="Design ids that should be discussed in this section, if applicable",
    )


class ReportBlueprintGroup(BaseModel):
    """A strategy group planned by the reporter."""

    group_id: str = Field(description="Short group identifier, e.g. A")
    group_name: str = Field(description="Strategy family name in the report language")
    design_purpose: str = Field(description="Main medicinal-chemistry purpose of this group")
    sar_output: str = Field(description="What SAR signal this group is expected to generate")
    section_intro: str = Field(description="Short paragraph to introduce this group in the HTML report")


class ReportBlueprintMolecule(BaseModel):
    """A molecule card planned by the reporter."""

    design_id: str = Field(description="Report design id, e.g. A1")
    asset_id: str = Field(description="Asset id supplied by code, e.g. M001")
    smiles: str = Field(description="Exact candidate SMILES from the input assets")
    strategy_group: str = Field(description="Group id or name")
    priority: Literal["H", "M", "L"] = Field(description="Priority: H, M, or L")
    role_tags: List[str] = Field(default_factory=list, description="Short visual tags for this design")
    design_logic: str = Field(description="Medicinal-chemistry design logic")
    sar_exploration_direction: str = Field(description="Expected SAR exploration direction")
    synthetic_feasibility: str = Field(description="Synthetic feasibility assessment")


class ReportBlueprint(BaseModel):
    """Agent-planned report structure before HTML authoring."""

    report_language: Literal["zh", "en"] = Field(
        default="zh",
        description="Report language: zh for Chinese, en for English",
    )
    report_title: str = Field(description="Report title in the report language")
    subtitle: str = Field(description="Short subtitle or metadata line in the report language")
    executive_recap: List[str] = Field(description="High-level recap bullets")
    design_philosophy: List[str] = Field(description="Main SAR/design lines to emphasize")
    original_compound_summary: str = Field(description="Original molecule analysis paragraph")
    sections: List[ReportBlueprintSection] = Field(description="Planned report sections")
    strategy_groups: List[ReportBlueprintGroup] = Field(description="Planned strategy groups")
    molecule_cards: List[ReportBlueprintMolecule] = Field(description="Detailed molecule cards to render")
    complete_candidate_smiles: List[str] = Field(description="All candidate SMILES that must appear in the report")
    first_round_synthesis: List[ReportFirstRoundCandidate] = Field(description="First-round synthesis recommendations")
    risk_notes: List[str] = Field(description="Risk notes and validation recommendations")
    expert_questions: List[str] = Field(
        default_factory=list,
        description="Questions the medicinal chemistry team should confirm after reading the report",
    )
    style_notes: List[str] = Field(
        default_factory=list,
        description="Visual and layout notes for the HTML writer",
    )

# ==================== Node 1: Information Preparation ====================

class ExtractInitialInfo(BaseModel):
    """Extracts initial information from user prompt."""
    smiles: str = Field(description="The SMILES of the Molecule to be optimized")
    optimization_goal: str = Field(description="Optimization Goal. Please keep it as concise as possible, no redundant content, only the core.")
    node_num_needed: Optional[int] = Field(
        default=None,
        description=(
            "The number of optimized candidate molecules required by the user. "
            "Leave null when the user does not specify a candidate count. "
            "Do not populate this from requested MCGS/search rounds or iterations."
        )
    )
    iteration_num_needed: Optional[int] = Field(
        None,
        description=(
            "The number of MCGS/search/optimization rounds or iterations explicitly requested by the user. "
            "Leave null when the user specifies only a candidate molecule count or does not specify rounds."
        ),
    )
    project_manager_brief: str = Field(
        "",
        description=(
            "A concise natural-language project-manager brief for downstream Rational, Creative, and Critic agents. "
            "Include only user-stated optimization goals, explicit constraints, preferences, allowed/disallowed changes, "
            "uploaded/document context cues, and explicitly requested user priorities. Do not add inferred secondary "
            "property objectives such as logP, solubility, TPSA, hERG, clearance, permeability, or synthetic feasibility "
            "unless the user explicitly requested them or they are hard constraints in uploaded/table context. "
            "Do not include hidden reasoning."
        ),
    )
    protein_sequence: Optional[str] = Field(None, description="The protein sequence corresponding to the molecule to be optimized")

# ==================== Node 3: Candidate Selection ====================

class SelectPromisingMolecule(BaseModel):
    """Result of selecting the most promising molecule from candidates."""
    reason: str = Field(description="The reason you chose this molecule as the next graph node. Please keep it as concise as possible, no redundant content, only the core.")
    smiles: str = Field(description="The SMILES of the most promising Molecule")
    selection_context: str = Field(description="Concise context note for the chosen molecule. Do not propose future optimization directions, next-step design strategies, or modification plans.")


class SharedMoleculeAnalysis(BaseModel):
    """Compact shared analysis for the current graph molecule."""

    shared_analysis_summary: str = Field(
        description="One concise summary of the current molecule's key optimization issue and opportunity."
    )
    shared_keep_fragments: List[str] = Field(
        default_factory=list,
        description="Fragments or molecular features downstream agents should try to preserve.",
    )
    shared_modifiable_fragments: List[str] = Field(
        default_factory=list,
        description="Fragments, positions, or features that are reasonable to modify.",
    )
    shared_risk_alerts: List[str] = Field(
        default_factory=list,
        description="Important liabilities or risks to keep in mind.",
    )
    shared_priority_directions: List[str] = Field(
        default_factory=list,
        description="Short priority directions for the next generation/evaluation step.",
    )


class SelectedADMETProperty(BaseModel):
    """One task-level ADMET endpoint selected for fixed filtering/evaluation."""

    property_name: str = Field(
        description="Exact ADMET endpoint name from the available property metadata."
    )
    direction: Literal["higher", "lower"] = Field(
        description="Tool preference direction for this endpoint. Must be exactly higher or lower."
    )
    rationale: str = Field(
        "",
        description="Brief reason this endpoint is directly relevant to the optimization task."
    )

    @field_validator("property_name", mode="before")
    @classmethod
    def _normalize_property_name(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Any) -> str:
        return str(value or "").strip().lower()


class ADMETPropertySelection(BaseModel):
    """Structured LLM output for the task-level frozen ADMET policy."""

    selected_properties: List[SelectedADMETProperty] = Field(
        default_factory=list,
        description="Zero to four directly task-relevant ADMET endpoints."
    )
    rationale: str = Field(
        "",
        description="Concise explanation of why these endpoints are the frozen task-level ADMET policy."
    )

    @field_validator("selected_properties")
    @classmethod
    def _limit_selected_properties(
        cls,
        value: List[SelectedADMETProperty],
    ) -> List[SelectedADMETProperty]:
        if len(value) > 4:
            raise ValueError("Select at most four ADMET endpoints.")
        return value
