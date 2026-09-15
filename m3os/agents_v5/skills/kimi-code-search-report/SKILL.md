---
name: kimi-code-search-report
description: Turn an M3OS molecular search graph into a polished, self-contained HTML decision report
type: inline
whenToUse: When report_input.json and molecule_assets.json are present in the report workspace
disableModelInvocation: false
---

# M3OS optimization report

Create the requested `report.html` from `report_input.json`,
`molecule_assets.json`, and the prebuilt graph component named by
`coding_runtime.graph_component_file`. The search graph is the sole source of molecular and
optimization claims. Do not invent molecules, measurements, citations,
properties, routes, or graph relationships. Label computed values and agent
judgments as such; show missing evidence as `N/A` or an explicit uncertainty.
Derive all counts, rankings, round summaries, and parent relationships from the
input data in code; do not hard-code statistics into the report narrative.
Write the report in `report_request.language`, including its title, headings,
controls, explanations, and recommendation copy; supplied English chemical
actions/rationales may remain verbatim. Set the matching HTML `lang` attribute.

## Report requirements

- Give an answer-first decision summary, root profile, interactive evolution
  graph, branch/round analysis, focused candidate cards, one compact full-node
  audit/comparison table, synthesis shortlist, risks, and next experiments.
- Treat `coding_runtime.graph_component_file` as an opaque, finished component.
  In the report build script, load its complete UTF-8 text from disk and insert
  that string exactly once at the evolution-graph position. Do not read it into
  the model context, rewrite it, reformat it, minify it, extract its CSS or
  JavaScript, or build another graph. The component already implements the
  authoritative M3OS frontend layout, complete node inspector, enlarged click
  targets, zoom/pan/drag/Fit interactions, responsive styling, and print rules.
  The final self-contained HTML must contain the prebuilt component text
  verbatim. If the file is absent, stop instead of substituting another graph.
  [references/molecular-search-graph.md](references/molecular-search-graph.md)
  is for maintaining the deterministic renderer, not required reading during
  ordinary report generation.
- Use the supplied `molecule_number` and `molecule_label` as the primary
  human-facing molecule identity throughout the report. Never derive, reorder,
  or renumber molecules yourself. Use `Molecule N` in card titles,
  recommendations, tiers, risk summaries, synthesis rows, next experiments,
  graph details, and parent/child labels. Keep full node IDs only in `data-*`
  attributes and internal JavaScript state; never show them as visible labels.
  When prose or a compact list refers to a molecule, wrap the visible
  `Molecule N` label in an element with
  `data-molecule-ref="<full node id>"` and `data-molecule-number="N"`.
- The Full-Node Audit is the authoritative molecule-number lookup. Include
  every graph node exactly once, including the root, in ascending
  `molecule_number` order. Its first columns must show `Molecule N`, the compact
  topology, and the full canonical SMILES so a reader can resolve any numbered
  reference. Show a full copyable SMILES on every focused card and every audit
  row as secondary chemical data, not as the primary identity. A focused card
  must use `data-candidate-card="<full node id>"` and
  `data-molecule-number="N"`; an audit row must use
  `data-audit="<full node id>"` and `data-molecule-number="N"`.
  Never visually truncate a SMILES, replace its tail with `...`/`…`, use
  `text-overflow: ellipsis`, or hide it behind hover text. Wrap the complete
  visible string across lines with `white-space: normal`,
  `overflow-wrap: anywhere`, and/or `word-break: break-all`. Mark each complete
  full-SMILES element `data-full-smiles="<full node id>"`.
- Treat iteration 0 as the root baseline, not an optimization round. Use
  `search_graph_semantics.optimization_round_count` for headline round counts,
  and emit it as `<meta name="m3os-optimization-round-count" content="N">`.
- Render exactly `report_request.detailed_card_limit` detailed candidate cards.
  Preserve every graph molecule in one table marked `data-full-node-audit`.
  Do not add expanded per-node audit articles or
  repeat all molecule SVG pairs below the focused cards. Each audit row must
  include the child SMILES, direct-parent SMILES, round, edit, recommendation,
  scores/properties, and supplied evidence or risk text. The root-baseline row
  must be first, with its parent fields marked N/A.
  Add a compact unhighlighted topology thumbnail to every audit row using the
  node's `topology_svg` from `molecule_assets.json`; mark its container
  `data-audit-topology="<full node id>"`. Keep thumbnails small and let the
  table scroll inside its container.
- Put the focused cards in one grid marked
  `data-candidate-grid data-desktop-columns="3"`. Use exactly three columns at
  desktop and wider sizes—never four—then two and one columns at narrower
  breakpoints. Use `minmax(0, 1fr)` tracks so card contents cannot force a page
  overflow.
- Outside the prebuilt Molecular Search Graph component, every rendered
  non-root molecular depiction must show a labeled parent →
  child pair from one direct-parent comparison in `molecule_assets.json`: the unhighlighted
  `parent_svg` beside the child `svg`, whose orange marks structural changes and
  purple marks stereochemical changes. Wrap the pair in
  `data-comparison-node="<full child id>"`; mark its two visual containers with
  `data-parent-node="<full parent id>"` and `data-diff-node="<full child id>"`.
  Label both sides primarily as `Molecule P → Molecule C`; show each full
  SMILES beneath its depiction for chemical traceability, and state which
  direct parent is shown when a node has multiple parents. The root profile
  uses the unhighlighted root SVG.
- Candidate cards must make the parent → child edit, score/property deltas,
  supplied evidence, benefits, risks, and recommendation easy to scan.
- Do not render all RDKit descriptors as one wide single-row table inside a
  candidate card. Use a wrapping property grid/list marked
  `data-property-grid="<full node id>" data-property-columns="2"`. Candidate
  cards can be narrow when three cards share a row, so use at most two
  property cells per row and allow each property's label and value to stack
  vertically. Give grid tracks and cells `min-width: 0`, and wrap long values;
  no property grid or cell may have `scrollWidth > clientWidth` at the widest
  multi-card layout.
- Compute recommendation tiers, shortlists, summaries, and next-step tier
  references from one shared data structure. Do not hand-write a tier elsewhere
  in the narrative. Assert internally that every repeated tier/reference agrees.
- Make recommendation tiers and the synthesis shortlist a stacked,
  full-width section marked `data-recommendation-layout="stacked-full-width"`.
  Put the three tier panels in an equal-width grid marked
  `data-tier-summary-grid data-desktop-columns="3"`; mark the panels
  `data-recommendation-tier="A"`, `"B"`, and `"C"`. Place the synthesis
  shortlist below that grid—not beside it—as a full-width responsive table
  marked `data-synthesis-shortlist`. Mark each body row
  `data-synthesis-candidate="<full node id>" data-molecule-number="N"`, use
  `Molecule N` as the primary molecule label, and include the edit, key scores,
  and concise selection rationale. Full SMILES is optional here because the
  Full-Node Audit is the canonical lookup. The table may scroll inside its own
  container on narrow screens; do not turn it into a narrow, tall ordered list.
- Select the synthesis shortlist from all non-root candidates across the
  complete search graph. MCGS iteration is provenance only and must never be an
  eligibility filter or preference by itself. Rank candidates by the task
  objectives, supplied computed properties and scores, critic evidence,
  medicinal-chemistry risk, synthetic feasibility, and SAR information value.
- Do not infer risk categories with raw substring matching over rationale text:
  phrases such as "no basic N" or "low toxicity risk" must not become positive
  risk flags. Prefer quoting the supplied critic rationale; if risks are grouped,
  use negation-aware logic and verify every hit. Mark each grouped container with
  `data-risk-theme="<theme>"` and list each `Molecule N` at most once per theme,
  using the standard `data-molecule-ref` hook, even when its rationale matches
  multiple keywords in that theme.
- Support search/filtering, candidate checkboxes, per-candidate reviewer notes,
  CSV export, and export of a self-contained HTML that preserves selections and
  notes for the next reviewer. When the same candidate has controls in both a
  focused card and the audit table, changing either checkbox or note must update
  every visible control for that candidate immediately; do not wait for a page
  reload to reconcile duplicate controls. Persist that review state as JSON in the
  `data-state` attribute of `#saved-review-state`, not inside script text, so
  arbitrary reviewer notes cannot break a re-exported page. Pass raw
  `JSON.stringify(state)` to `setAttribute`; DOM serialization performs the
  necessary attribute escaping, so never pre-escape it with `&quot;`/`&amp;`.
  Likewise, pass raw note text into the CSV row and escape every CSV cell exactly
  once in the shared serializer; do not pre-double quotes in the note and then
  double them again when serializing the row.

## Product and visual quality

Build an original, premium scientific decision document—not a generic dashboard
or a clone of another report. Use strong editorial hierarchy, restrained color,
generous whitespace, aligned molecular drawings, compact evidence tables, clear
states, responsive layouts, and print styling. The evolution graph and molecular
changes should be the dominant visual narrative; avoid decorative effects that
do not improve scientific review.

At desktop width, use the available space as a designed composition: make the
first screen a concise executive overview, keep the graph visually prominent,
and keep candidate cards at exactly three columns even on very wide screens,
then reduce to two and one columns responsively. A centered content width around
1320–1440 px gives the complex parent → child cards enough room without making
the page feel sparse. Grid children must be allowed to shrink (`min-width: 0`),
and wide graphs/tables must scroll inside their own containers; the page itself
must not overflow horizontally at 1024 px or 680 px. At the one-column
breakpoint, override any desktop `grid-column: span N` rules so every
summary/root panel occupies the same full width. Keep recommendation tiers in
three useful-width columns and their synthesis table on a separate full-width
row. Avoid a long sequence of full-width form-like rows, tiny typography,
oversized empty image areas, and dense prose.

The output must be one substantial HTML file with inline CSS, JavaScript, SVG,
and image data only. It must remain useful without JavaScript and must not use
CDNs, remote fonts, remote images, network calls, or sibling assets. Preserve
every candidate SMILES verbatim in the complete audit and validate the final
HTML and JavaScript before finishing. Make embedded SVG responsive with CSS;
never use a non-length value such as `height="auto"` as an SVG attribute.
Properties in `search_graph` are supplied search values; descriptors in
`molecule_assets.json` are deterministic RDKit calculations. Label those
provenances separately, and do not propose experimentally measuring or
confirming a deterministic descriptor such as HBA count. Use the supplied
`molecule_assets.root.descriptors` in the root profile rather than reporting
them as unavailable.
Generate repeated inline markup with a helper/loop rather than hand-joining tag
fragments, and check that paired tags such as `<code>` are balanced. The final
UTF-8 document must not contain Unicode replacement characters (`�`).
