# ADMET Filtering Rules

Use this reference when configuring `admet_filter_by_admetai` after generation.

## Core policy
- Filter only on the Frozen ADMET endpoints selected by the workflow for the current task.
- Do not add, remove, replace, or change frozen ADMET endpoints or directions.
- Pass the exact preceding-stage CSV path as `candidate_csv_path`; do not rely on an invented `rand_str` to infer the CSV path.
- Pass the exact `rand_str` returned by the generation setup tool. Never invent labels such as `bbbp_primary_candidates`.
- Before the tool call, self-check that `preference_json` exactly matches the Frozen ADMET preference_json and every value is exactly `"higher"` or `"lower"`.
- Treat metadata `Preference` values such as `Intermediate` and `Context_Specific` as descriptive defaults, not valid tool values. An explicit task request to increase or decrease the property maps to `"higher"` or `"lower"`, respectively.
- Do not map a target value, acceptable range, or "keep intermediate" objective to an arbitrary direction. The current filter ranks monotonically and cannot optimize distance to a target or range membership.
- If the frozen JSON is `{}`, do not call ADMET filtering and state that no directly relevant ADMET endpoint is available.
- Use `top_n=5` when ADMET is the final filter. Use `top_n=20` when Nesso target-activity filtering follows ADMET.
- Do not inspect the original generation CSV. Generated files live on the MCP server, so use the filter tool result instead of local `glob`, `ls`, or CSV reads.

## Good examples
- BBB penetration only: `{"BBB_Martins": "higher"}`.
- BBB plus hERG: `{"BBB_Martins": "higher", "hERG": "lower"}`.
- Solubility-only optimization: use the relevant solubility endpoint with either `"higher"` or `"lower"` according to the property definition.
- Explicitly lower an intermediate-preference property: `{"logP": "lower"}`.

## Bad examples
- `{"BBB_Martins": "higher_better"}`
- `{"BBB_Martins": "increase"}`
- `{"BBB_Martins": true}`
- `{"Lipophilicity_AstraZeneca": "intermediate"}`

## Missing endpoint handling
- If the requested endpoint is absent from the property metadata, state that limitation clearly.
- Do not substitute adjacent but different CYP or toxicity endpoints as if they answered the same question.
- Fall back to medicinal chemistry judgment for final triage when direct endpoint support is unavailable.

## Shortlist behavior
- Use filtering to prune obvious mismatches, not to remove all diversity.
- When ADMET is the final stage, its five returned candidates are the complete current-round set.
- When Nesso follows ADMET, pass all 20 ADMET candidates to Nesso and use Nesso's ten returned candidates as the complete current-round set.

[Note]: Use the Frozen ADMET preference_json from the current task data exactly. Do not privately select ADMET attributes.
