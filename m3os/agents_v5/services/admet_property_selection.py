"""Task-level ADMET endpoint selection and enforcement helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from langchain_core.messages import HumanMessage, SystemMessage

from m3os.agents_v5.core.config import AgentConfig
from m3os.agents_v5.core.models import (
    ADMETPropertySelection,
    SelectedADMETProperty,
)
from m3os.agents_v5.core.retry_utils import (
    invoke_with_retry,
    validate_structured_response,
)
from m3os.agents_v5.services.llm_factory import create_base_model


ADMET_SELECTION_SYSTEM_PROMPT = """You choose the frozen ADMET or physicochemical tool endpoints for one molecular optimization task.

Rules:
- Select only endpoints whose names exactly match the provided metadata.
- Select zero to four endpoints.
- Select endpoints only when they are the most direct available property representation of the user's optimization task.
- Do not add generic drug-likeness or safety endpoints unless the task explicitly asks for them.
- If the requested endpoint is unavailable, do not substitute a nearby endpoint.
- Metadata Preference values such as "Intermediate" or "Context_Specific" describe
  a general tendency; they are not legal tool directions. When the task explicitly
  asks to increase or decrease such a property, select "higher" or "lower"
  respectively, regardless of the metadata Preference.
- A target value, acceptable range, or request to keep a property intermediate is
  not expressible by these monotonic directions. Do not invent "higher" or "lower"
  unless the task also states a monotonic direction.
- For each selected endpoint, choose direction exactly "higher" or "lower" for the ADMET tools.
- Return an empty selected_properties list when no available tool endpoint is directly relevant.
"""

DEFAULT_ADMET_SELECTION_TIMEOUT_SECONDS = 25.0

LOWER_DIRECTION_TERMS = (
    "avoid",
    "decrease",
    "de-risk",
    "derisk",
    "less",
    "lower",
    "minimize",
    "reduce",
    "suppress",
    "降低",
    "减少",
    "减小",
    "减轻",
    "下降",
)
HIGHER_DIRECTION_TERMS = (
    "boost",
    "enhance",
    "higher",
    "improve",
    "increase",
    "maximize",
    "promote",
    "raise",
    "增加",
    "增强",
    "提高",
    "提升",
)
LOWER_BBB_TERMS = (
    "avoid bbb",
    "avoid brain penetration",
    "decrease bbb",
    "decrease blood brain barrier",
    "decrease brain penetration",
    "decrease brain permeability",
    "lower bbb",
    "lower blood brain barrier",
    "lower brain penetration",
    "lower brain permeability",
    "reduce bbb",
    "reduce blood brain barrier",
    "reduce brain penetration",
    "reduce brain permeability",
    "降低 血脑屏障",
    "减少 血脑屏障",
)
DIRECT_ENDPOINT_HINTS: Dict[str, Dict[str, Any]] = {
    "molecular_weight": {
        "terms": ("molecular weight", "mol wt", "molwt", "mw", "分子量"),
        "rationale": "The task directly asks for molecular weight.",
    },
    "logP": {
        "terms": (
            "logp",
            "log p",
            "clogp",
            "c logp",
            "octanol water partition coefficient",
        ),
        "rationale": "The task directly asks for RDKit MolLogP.",
    },
    "hydrogen_bond_acceptors": {
        "terms": ("hydrogen bond acceptors", "hydrogen bond acceptor count", "hba", "氢键受体"),
        "rationale": "The task directly asks for hydrogen-bond acceptor count.",
    },
    "hydrogen_bond_donors": {
        "terms": ("hydrogen bond donors", "hydrogen bond donor count", "hbd", "氢键供体"),
        "rationale": "The task directly asks for hydrogen-bond donor count.",
    },
    "Lipinski": {
        "terms": ("lipinski", "rule of five", "rule-of-five", "ro5", "利平斯基"),
        "default_direction": "higher",
        "rationale": "The task directly asks for Lipinski rule-of-five compliance.",
    },
    "QED": {
        "terms": (
            "qed",
            "quantitative estimate of drug likeness",
            "quantitative estimate of drug-likeness",
            "类药性",
        ),
        "default_direction": "higher",
        "rationale": "The task directly asks for QED drug-likeness.",
    },
    "stereo_centers": {
        "terms": ("stereocenters", "stereo centers", "stereocenter count", "立体中心"),
        "rationale": "The task directly asks for stereocenter count.",
    },
    "tpsa": {
        "terms": ("tpsa", "topological polar surface area", "拓扑极性表面积"),
        "rationale": "The task directly asks for topological polar surface area.",
    },
    "BBB_Martins": {
        "terms": (
            "bbb",
            "bbbp",
            "blood brain barrier",
            "blood-brain barrier",
            "brain barrier",
            "brain penetration",
            "brain permeability",
            "cns penetration",
            "cns permeability",
            "血脑屏障",
        ),
        "default_direction": "higher",
        "rationale": "The task directly asks for blood-brain barrier permeability.",
    },
    "hERG": {
        "terms": (
            "herg",
            "herg blocking",
            "herg inhibition",
            "herg liability",
            "qt prolongation",
        ),
        "default_direction": "lower",
        "rationale": "The task directly asks to reduce hERG blocking/liability.",
    },
    "Solubility_AqSolDB": {
        "terms": ("aqueous solubility", "solubility", "水溶性", "溶解度"),
        "default_direction": "higher",
        "rationale": "The task directly asks for aqueous solubility.",
    },
    "Bioavailability_Ma": {
        "terms": ("oral bioavailability", "bioavailability", "口服生物利用度", "生物利用度"),
        "default_direction": "higher",
        "rationale": "The task directly asks for oral bioavailability.",
    },
    "HIA_Hou": {
        "terms": ("human intestinal absorption", "intestinal absorption", "肠道吸收"),
        "default_direction": "higher",
        "rationale": "The task directly asks for human intestinal absorption.",
    },
    "Caco2_Wang": {
        "terms": ("caco2", "caco 2", "caco-2"),
        "default_direction": "higher",
        "rationale": "The task directly asks for Caco-2 permeability.",
    },
    "PAMPA_NCATS": {
        "terms": ("pampa",),
        "default_direction": "higher",
        "rationale": "The task directly asks for PAMPA permeability.",
    },
    "Pgp_Broccatelli": {
        "terms": (
            "p glycoprotein inhibition",
            "p-glycoprotein inhibition",
            "pgp inhibition",
            "p gp inhibition",
        ),
        "default_direction": "lower",
        "rationale": "The task directly asks for P-glycoprotein inhibition.",
    },
    "Half_Life_Obach": {
        "terms": ("half life", "half-life", "半衰期"),
        "default_direction": "higher",
        "rationale": "The task directly asks for half-life.",
    },
    "LD50_Zhu": {
        "terms": ("ld50", "acute toxicity", "急性毒性"),
        "default_direction": "lower",
        "rationale": "The task directly asks for acute toxicity/LD50.",
    },
    "AMES": {
        "terms": ("ames", "mutagenicity", "致突变"),
        "default_direction": "lower",
        "rationale": "The task directly asks for mutagenicity/AMES risk.",
    },
    "DILI": {
        "terms": ("dili", "drug induced liver injury", "drug-induced liver injury", "肝损伤"),
        "default_direction": "lower",
        "rationale": "The task directly asks for drug-induced liver injury.",
    },
    "ClinTox": {
        "terms": ("clintox", "clinical toxicity", "toxicity in clinical trials", "临床毒性"),
        "default_direction": "lower",
        "rationale": "The task directly asks for clinical toxicity.",
    },
    "Carcinogens_Lagunin": {
        "terms": ("carcinogen", "carcinogenicity", "致癌"),
        "default_direction": "lower",
        "rationale": "The task directly asks for carcinogenicity.",
    },
}


def canonical_admet_preference_json(preferences: Mapping[str, Any]) -> str:
    """Return stable JSON for an ADMET preference mapping."""
    normalized = {
        str(prop).strip(): str(direction).strip().lower()
        for prop, direction in preferences.items()
        if str(prop).strip()
    }
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def parse_admet_preference_json(
    preference_json: Any,
    *,
    require_non_empty: bool,
) -> Dict[str, str]:
    """Parse and normalize an ADMET preference JSON object."""
    if isinstance(preference_json, Mapping):
        raw_preferences = dict(preference_json)
    else:
        try:
            raw_preferences = json.loads(str(preference_json or ""))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("preference_json must be a JSON object string.") from exc
    if not isinstance(raw_preferences, dict):
        raise ValueError("preference_json must decode to a JSON object.")

    normalized: Dict[str, str] = {}
    invalid: Dict[str, Any] = {}
    for prop, direction in raw_preferences.items():
        key = str(prop).strip()
        if not key:
            continue
        value = str(direction).strip().lower()
        if value not in {"higher", "lower"}:
            invalid[key] = direction
            continue
        normalized[key] = value

    if invalid:
        invalid_text = ", ".join(f"{key}={value!r}" for key, value in invalid.items())
        raise ValueError(
            "Invalid preference_json direction(s): "
            f"{invalid_text}. Values must be exactly \"higher\" or \"lower\"."
        )
    if require_non_empty and not normalized:
        raise ValueError("preference_json must contain at least one ADMET endpoint.")
    return normalized


def enforce_frozen_admet_preference_json(
    submitted_preference_json: Any,
    frozen_preference_json: Any,
    *,
    tool_name: str,
) -> str:
    """Require a tool call to use exactly the frozen task-level ADMET policy."""
    if frozen_preference_json is None or str(frozen_preference_json).strip() == "":
        return canonical_admet_preference_json(
            parse_admet_preference_json(submitted_preference_json, require_non_empty=True)
        )

    expected = parse_admet_preference_json(frozen_preference_json, require_non_empty=False)
    if not expected:
        raise ValueError(
            f"{tool_name} is not allowed for this task because no directly relevant "
            "ADMET endpoint was selected by the workflow."
        )

    actual = parse_admet_preference_json(submitted_preference_json, require_non_empty=True)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    wrong_direction = sorted(
        prop for prop in set(expected) & set(actual)
        if expected[prop] != actual[prop]
    )
    if missing or extra or wrong_direction:
        details: List[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        if wrong_direction:
            details.append(
                "wrong_direction="
                + str({prop: {"expected": expected[prop], "got": actual[prop]} for prop in wrong_direction})
            )
        raise ValueError(
            f"{tool_name} must use the frozen task-level ADMET preference_json exactly. "
            f"Expected {canonical_admet_preference_json(expected)}, got "
            f"{canonical_admet_preference_json(actual)}. {'; '.join(details)}."
        )
    return canonical_admet_preference_json(expected)


def build_selected_admet_prompt_block(state: Mapping[str, Any]) -> str:
    """Format frozen ADMET policy for agent user prompts."""
    preference_json = str(state.get("selected_admet_preference_json") or "{}")
    properties = state.get("selected_admet_properties") or []
    details = state.get("selected_admet_property_details") or []
    rationale = str(state.get("selected_admet_property_rationale") or "")
    if not properties:
        return (
            "\n\n## [TASK-LEVEL FROZEN ADMET POLICY]\n"
            "* **Frozen ADMET preference_json:** {}\n"
            "* **Frozen ADMET properties:** []\n"
            "* **Policy:** No directly relevant ADMET endpoint was selected. "
            "Do not call `admet_filter_by_admetai` or `admet_predict_by_admetai` for this task; "
            "state the limitation and use medicinal chemistry judgment instead.\n"
        )
    return (
        "\n\n## [TASK-LEVEL FROZEN ADMET POLICY]\n"
        f"* **Frozen ADMET preference_json:** {preference_json}\n"
        f"* **Frozen ADMET properties:** {json.dumps(properties, ensure_ascii=False)}\n"
        f"* **Frozen ADMET details:** {json.dumps(details, ensure_ascii=False)}\n"
        f"* **Selection rationale:** {rationale}\n"
        "* **Mandatory rule:** Whenever calling `admet_filter_by_admetai` or "
        "`admet_predict_by_admetai`, pass the exact Frozen ADMET preference_json above. "
        "Do not add, remove, replace, or change directions for ADMET properties.\n"
    )


class ADMETPropertySelectionService:
    """Select and validate the fixed ADMET policy for one optimization task."""

    def __init__(self, config: AgentConfig):
        self.config = config

    async def select_for_task(
        self,
        *,
        initial_smiles: str,
        initial_iupac: str,
        initial_fragments: str,
        optimization_goal: str,
        project_manager_brief: Any,
        protein_sequence: Any,
    ) -> Dict[str, Any]:
        property_lookup = self.load_property_metadata()
        task_context = {
            "initial_smiles": initial_smiles,
            "optimization_goal": optimization_goal,
            "project_manager_brief": project_manager_brief,
            "protein_sequence_present": bool(protein_sequence),
        }
        if self.config.enable_molecular_auxiliary_context:
            task_context.update(
                {
                    "initial_iupac": initial_iupac,
                    "initial_fragments": initial_fragments,
                }
            )
        task_signature = self.build_task_signature(task_context)
        if not property_lookup:
            return self.empty_selection_payload(
                task_signature,
                "No ADMET property metadata was available.",
            )

        # Rule-based keyword pre-selection is intentionally disabled. Keep the
        # legacy matcher below for reference, but let the structured LLM decide
        # the ADMET endpoint(s) and direction from the full task context.
        # direct_payload = self.build_direct_selection_payload(
        #     task_context,
        #     property_lookup=property_lookup,
        #     task_signature=task_signature,
        # )
        # if direct_payload["selected_admet_properties"]:
        #     print(
        #         "[ADMET selection] Rule-based direct match: "
        #         f"{direct_payload.get('selected_admet_preference_json')}"
        #     )
        #     return direct_payload

        try:
            llm = create_base_model(self.config.llm)
            structured_llm = llm.with_structured_output(ADMETPropertySelection, include_raw=True)
            messages = [
                SystemMessage(content=ADMET_SELECTION_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        "[OPTIMIZATION TASK]\n"
                        f"{json.dumps(task_context, ensure_ascii=False, indent=2, default=str)}\n\n"
                        "[AVAILABLE ADMET/PHYSICOCHEMICAL ENDPOINTS]\n"
                        f"{json.dumps(list(property_lookup.values()), ensure_ascii=False, indent=2)}"
                    )
                ),
            ]

            async def _invoke():
                return await structured_llm.ainvoke(messages)

            timeout_seconds = self.selection_timeout_seconds()
            print(
                "[ADMET selection] Selecting task-level ADMET endpoints..."
                f" timeout={timeout_seconds:g}s"
            )
            call = invoke_with_retry(
                _invoke,
                max_retries=3,
                base_delay=1.0,
                validate_fn=validate_structured_response,
                operation_name="Select task-level ADMET properties LLM call",
            )
            if timeout_seconds > 0:
                response = await asyncio.wait_for(call, timeout=timeout_seconds)
            else:
                response = await call
            selection = response["parsed"] if isinstance(response, dict) else response
            payload = self.build_selection_payload(
                selection,
                property_lookup=property_lookup,
                task_signature=task_signature,
            )
            print(
                "[ADMET selection] Completed: "
                f"{payload.get('selected_admet_preference_json')}"
            )
            return payload
        except Exception as exc:
            print(
                "[ADMET selection] Failed; continuing with empty ADMET policy: "
                f"{type(exc).__name__}: {exc}"
            )
            return self.empty_selection_payload(
                task_signature,
                "ADMET endpoint selection failed or timed out, and no direct metadata endpoint "
                "was matched by the deterministic selector; continuing without ADMET filtering.",
            )

    def load_property_metadata(self) -> Dict[str, Dict[str, Any]]:
        path = self._property_meta_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}

        lookup: Dict[str, Dict[str, Any]] = {}
        for category_name, category in data.items():
            if not isinstance(category, dict):
                continue
            for property_name, meta in category.items():
                if not isinstance(meta, dict):
                    continue
                lookup[str(property_name)] = {
                    "name": str(property_name),
                    "category": str(category_name),
                    "description": str(meta.get("Description") or ""),
                    "units": str(meta.get("Units") or ""),
                    "task_type": str(meta.get("Task_Type") or ""),
                    "preference": str(meta.get("Preference") or ""),
                }
        return lookup

    def build_selection_payload(
        self,
        selection: ADMETPropertySelection | Mapping[str, Any],
        *,
        property_lookup: Mapping[str, Mapping[str, Any]],
        task_signature: str,
    ) -> Dict[str, Any]:
        selected_items = self._selection_items(selection)
        seen: set[str] = set()
        preferences: Dict[str, str] = {}
        details: List[Dict[str, Any]] = []

        for item in selected_items:
            name = item.property_name
            if not name:
                continue
            if name in seen:
                raise ValueError(f"Duplicate ADMET endpoint selected: {name}")
            if name not in property_lookup:
                raise ValueError(f"Unknown ADMET endpoint selected: {name}")
            direction = str(item.direction or "").strip().lower()
            if direction not in {"higher", "lower"}:
                raise ValueError(f"Invalid ADMET direction for {name}: {item.direction!r}")
            seen.add(name)
            preferences[name] = direction
            meta = dict(property_lookup[name])
            details.append(
                {
                    **meta,
                    "direction": direction,
                    "rationale": str(item.rationale or "").strip(),
                }
            )

        if len(details) > 4:
            raise ValueError("Select at most four ADMET endpoints.")

        rationale = (
            str(selection.rationale or "").strip()
            if isinstance(selection, ADMETPropertySelection)
            else str(selection.get("rationale") or "").strip()
        )
        if not details and not rationale:
            rationale = "No available ADMET endpoint directly matches the optimization task."
        return {
            "selected_admet_properties": list(preferences),
            "selected_admet_preference_json": canonical_admet_preference_json(preferences),
            "selected_admet_property_details": details,
            "selected_admet_property_rationale": rationale,
            "selected_admet_task_signature": task_signature,
        }

    def build_direct_selection_payload(
        self,
        task_context: Mapping[str, Any],
        *,
        property_lookup: Mapping[str, Mapping[str, Any]],
        task_signature: str,
    ) -> Dict[str, Any]:
        """Select endpoints only when task text directly names available metadata."""
        task_text = self._normalized_task_text(task_context)
        selected: List[SelectedADMETProperty] = []
        seen: set[str] = set()

        for property_name, meta in property_lookup.items():
            if len(selected) >= 4:
                break
            match_reason = self._direct_match_reason(property_name, meta, task_text)
            if not match_reason or property_name in seen:
                continue
            direction = self._direction_for_direct_match(property_name, meta, task_text)
            if direction not in {"higher", "lower"}:
                continue
            selected.append(
                SelectedADMETProperty(
                    property_name=property_name,
                    direction=direction,
                    rationale=match_reason,
                )
            )
            seen.add(property_name)

        if not selected:
            return self.empty_selection_payload(
                task_signature,
                "No available ADMET endpoint directly matches the optimization task.",
            )

        rationale = "Deterministic direct metadata match for explicitly requested ADMET endpoint(s)."
        return self.build_selection_payload(
            ADMETPropertySelection(
                selected_properties=selected,
                rationale=rationale,
            ),
            property_lookup=property_lookup,
            task_signature=task_signature,
        )

    @staticmethod
    def empty_selection_payload(task_signature: str, rationale: str) -> Dict[str, Any]:
        return {
            "selected_admet_properties": [],
            "selected_admet_preference_json": "{}",
            "selected_admet_property_details": [],
            "selected_admet_property_rationale": rationale,
            "selected_admet_task_signature": task_signature,
        }

    @staticmethod
    def build_task_signature(task_context: Mapping[str, Any]) -> str:
        text = json.dumps(task_context, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def selection_timeout_seconds() -> float:
        raw_value = os.getenv("M3OS_ADMET_SELECTION_TIMEOUT_SECONDS")
        if raw_value is None:
            return DEFAULT_ADMET_SELECTION_TIMEOUT_SECONDS
        try:
            return max(0.0, float(raw_value))
        except (TypeError, ValueError):
            return DEFAULT_ADMET_SELECTION_TIMEOUT_SECONDS

    def _property_meta_path(self) -> Path:
        configured = getattr(self.config.paths, "property_meta_info_path", None)
        if configured:
            return Path(str(configured)).expanduser()
        return Path(__file__).resolve().parents[1] / "tools" / "property_meta_info_preference.json"

    @classmethod
    def _normalized_task_text(cls, task_context: Mapping[str, Any]) -> str:
        raw_text = json.dumps(task_context, ensure_ascii=False, sort_keys=True, default=str)
        return cls._normalize_for_match(raw_text)

    @staticmethod
    def _normalize_for_match(value: Any) -> str:
        text = str(value or "").casefold()
        text = text.replace("β", "beta")
        text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
        return f" {' '.join(text.split())} "

    @classmethod
    def _contains_phrase(cls, task_text: str, phrase: Any) -> bool:
        normalized = cls._normalize_for_match(phrase).strip()
        if not normalized:
            return False
        return f" {normalized} " in task_text

    @classmethod
    def _direct_match_reason(
        cls,
        property_name: str,
        meta: Mapping[str, Any],
        task_text: str,
    ) -> str:
        if cls._contains_phrase(task_text, property_name):
            return f"The task explicitly names available ADMET endpoint {property_name}."

        hint = DIRECT_ENDPOINT_HINTS.get(property_name)
        if hint:
            for term in hint.get("terms", ()):
                if cls._contains_phrase(task_text, term):
                    return str(hint.get("rationale") or f"The task directly asks for {property_name}.")

        description = str(meta.get("description") or "")
        if len(cls._normalize_for_match(description).strip().split()) >= 2 and cls._contains_phrase(task_text, description):
            return f"The task directly matches metadata description for {property_name}."
        return ""

    @classmethod
    def _direction_for_direct_match(
        cls,
        property_name: str,
        meta: Mapping[str, Any],
        task_text: str,
    ) -> str:
        if property_name == "hERG":
            return "lower"
        if property_name == "BBB_Martins":
            return "lower" if cls._has_any_term(task_text, LOWER_BBB_TERMS) else "higher"

        hint = DIRECT_ENDPOINT_HINTS.get(property_name)
        if hint and hint.get("default_direction") in {"higher", "lower"}:
            return str(hint["default_direction"])

        preference = str(meta.get("preference") or "").strip().casefold()
        if preference in {"higher", "true"}:
            return "higher"
        if preference in {"lower", "false"}:
            return "lower"

        if cls._has_any_term(task_text, LOWER_DIRECTION_TERMS):
            return "lower"
        if cls._has_any_term(task_text, HIGHER_DIRECTION_TERMS):
            return "higher"
        return ""

    @classmethod
    def _has_any_term(cls, task_text: str, terms: tuple[str, ...]) -> bool:
        return any(cls._contains_phrase(task_text, term) for term in terms)

    @staticmethod
    def _selection_items(
        selection: ADMETPropertySelection | Mapping[str, Any],
    ) -> List[SelectedADMETProperty]:
        if isinstance(selection, ADMETPropertySelection):
            return list(selection.selected_properties or [])
        raw_items = selection.get("selected_properties") if isinstance(selection, Mapping) else []
        if not isinstance(raw_items, list):
            raise ValueError("selected_properties must be a list.")
        return [SelectedADMETProperty(**dict(item)) for item in raw_items if isinstance(item, Mapping)]
