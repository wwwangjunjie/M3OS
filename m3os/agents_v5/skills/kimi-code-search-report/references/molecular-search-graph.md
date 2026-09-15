# M3OS Molecular Search Graph

Use this component contract for the report's evolution graph. It is adapted
from M3OS Server's `Molecular Search Graph` frontend, not from the example
reports. Preserve its information architecture, layout semantics, colors, and
interactions while integrating the typography and spacing with the surrounding
report.

## Required composition

Render one prominent inspector section with:

1. A header containing `Molecular Search Graph` (translated when the report is
   Chinese) and compact chips for node count, edge count, best score, and round
   count.
2. A two-column inspector at desktop width: the graph stage on the left and a
   selected-node detail panel on the right. Stack them on narrow screens.
3. A graph shell with `+`, `-`, and `Fit` controls and a legend below the SVG.
4. A detail panel that updates when a graph node is clicked or activated with
   Enter/Space. Show the selected molecule topology, full SMILES, score, round,
   visits/UCT when available, direct parent SMILES, edit, supplied generator and
   critic rationales, and compact properties. Use the supplied `Molecule N`
   identity and never require the reader to decode a node ID.

The report may additionally cross-highlight or scroll to a candidate card, but
that must not replace the selected-node detail panel.

## DOM contract

Use these stable hooks so the host validator can distinguish this component
from a generic chart:

```html
<section data-evolution-graph="m3os-molecular-search-graph-v1">
  <div data-graph-metrics>...</div>
  <div data-graph-inspector>
    <div data-graph-stage>
      <div data-graph-controls>
        <button data-graph-zoom="in">+</button>
        <button data-graph-zoom="out">-</button>
        <button data-graph-fit>Fit</button>
      </div>
      <svg data-graph-svg data-home-viewbox="...">
        <g data-graph-panzoom>
          <g data-graph-layer="0">...</g>
          <g data-graph-edge="SOURCE_ID|TARGET_ID">...</g>
          <g data-graph-node="FULL_NODE_ID" data-molecule-number="N"
             role="button" tabindex="0">
            <circle data-graph-node-hit="FULL_NODE_ID">...</circle>
            ...
          </g>
        </g>
      </svg>
      <div data-graph-legend>...</div>
    </div>
    <div data-graph-node-details>
      <article data-graph-node-detail="FULL_NODE_ID"
               data-molecule-number="N">...</article>
      ...
    </div>
  </div>
</section>
```

Emit exactly one `data-graph-node` and one `data-graph-node-detail` for every
node in `search_graph.nodes`, including the root. Emit exactly one
`data-graph-edge` for every valid source/target relationship in
`search_graph.edges`. A node/detail may be hidden when inactive but must remain
in the document. Use complete IDs only in data attributes. The compact graph
circle displays the supplied molecule number; its detail heading displays
`Molecule N`. Show `Root`, `Best`, and `Current` only as status indicators, not
as molecule identities. The full SMILES appears in the detail as secondary
chemical data. Direct parents are identified primarily as `Molecule P`, with
their full SMILES beneath the numbered reference.
Every node group must be keyboard focusable with `role="button"` and contain
exactly one transparent `data-graph-node-hit="FULL_NODE_ID"` circle that is
larger than the visible node.

## Layered layout

This is a parent/generation search graph, not a score scatterplot. Never place
nodes vertically according to score, never draw a numeric score Y axis, and do
not collapse a round into overlapping points.

Reproduce the frontend's deterministic layout:

- Find the root (`is_root`, otherwise the first node).
- Build parent/child adjacency from valid edges.
- Compute the minimum graph depth of every reachable node with breadth-first
  traversal from the root. For an unreachable node, fall back to iteration (at
  least layer 1). Depth, not raw iteration alone, determines the horizontal
  layer.
- Sort nodes by depth, then iteration, then stable ID.
- Within each non-root layer, group nodes by iteration and their nearest parent
  in a previous layer. Order groups by iteration, then the parent's vertical
  position, then parent ID. Within a group, order by score descending then ID.
- Use at most three columns per parent group: one column for one child, two for
  two to four children, three for larger groups. Stack parent groups vertically
  with visible separation.
- Suggested geometry from the frontend: left `120`, top `94`, base layer gap
  `220`, child column gap `72`, row gap `88`, parent-group gap `62`; minimum
  canvas `920 × 560`. Increase layer separation when a layer is wide.
- Draw a softly tinted rounded background band for every layer and label it
  `Root`, `Layer 1`, `Layer 2`, etc.
- Draw directed straight edges behind nodes with arrowheads. Put a short edit or
  action label near the line midpoint, rotated to remain readable; retain the
  full action in a `<title>` or accessible label. Edge labels may be shortened,
  because full evidence is shown in the detail panel.

The layout must be computed from the JSON in JavaScript or in the report build
script. Do not hard-code coordinates for a particular example graph.

## Visual encoding

Match the M3OS frontend's restrained encoding:

- Root: pale blue fill, `#2563eb` stroke.
- Non-root palette by iteration, cycling through hues approximately
  `217, 174, 38, 262, 345, 190` with strong saturation.
- Within one iteration, darker fill means a higher finite score. Normalize only
  against that iteration's min/max so rounds remain visually distinct.
- Unreachable: muted slate fill/stroke and reduced opacity.
- Best: green `#10b981` stroke, about `3.5px`.
- Current: amber `#f59e0b` stroke, about `3.5px`.
- Best + current: dark stroke, about `4px`.
- Selected: dark `#111827` stroke, about `4px`, with a restrained shadow.
- Root/best/current radius about `28`; ordinary candidates about `23`.
- Show the supplied molecule number inside the circle and the formatted score
  immediately below it. The molecule drawing belongs in the detail panel, not
  inside every graph node.
- Legend: Root, iteration color, darker = higher score, Best, Current, and
  layered by generation/parent.

Keep the stage background near white with subtle blue/slate borders. The graph
must be visually dominant but not resemble a neon network diagram.

## Interaction

Implement the frontend behaviors without external libraries:

- Clicking or keyboard-activating a node marks it selected and switches the
  visible node detail.
- Make clicking discoverable and forgiving: apply `cursor: pointer` to the node
  group, show an obvious hover/focus ring, and put a transparent hit circle of
  about radius `34–40` behind the visible radius `23–28` circle. The hit circle
  must use `pointer-events: all`; labels remain `pointer-events: none`, so a
  pointer press anywhere around the visible node reliably targets the group.
- On selection, update `aria-pressed`/selected styling, reveal the matching
  detail, and reset the *outer* `[data-graph-node-details]` container to
  `scrollTop = 0`. Resetting only the selected article does not move the visible
  scroll position.
- `+` and `-` zoom around the center; the wheel zooms around the pointer.
- Pointer drag on empty SVG space pans the viewBox.
- `Fit` restores the original viewBox.
- Nodes can be dragged. Update their connected line endpoints and edge-label
  position/rotation immediately. Prevent node drag from also panning the graph.
  Capture the pointer on the node element itself, as the M3OS frontend does;
  do not capture a node press on the enclosing SVG, because that retargets the
  subsequent mouse click to the SVG and makes ordinary node clicks appear inert.
  Do not enter drag mode on pointer-down alone: require roughly 4–6 CSS pixels
  of movement. A press/release below that threshold must always select the node;
  after a true drag, suppress the synthetic click so it cannot undo or duplicate
  the gesture.
- Keep zoom bounds reasonable (the frontend uses roughly `360–1600` viewBox
  width and `180–900` height).

Initialize selection to best, then current, then root/first node. The report
must still show the complete graph and the initial selected detail when
JavaScript is unavailable; JavaScript enhances selection and navigation.

Before finishing, exercise at least two nodes with a real browser pointer click
on their visible/hit circles—not only `dispatchEvent`. Confirm that each click
changes both the selected ring and the sole visible matching detail. Also test a
small press/release motion and a true node drag to verify the movement threshold.

## Responsive and print behavior

- The graph/detail columns should both have `min-width: 0`.
- Give the SVG a responsive width and an aspect ratio near `1.64 / 1`, with a
  useful minimum height. At narrow widths, stack the graph above the details.
- Any unavoidable graph overflow stays inside the graph stage; the page itself
  must not overflow horizontally.
- For print, show the fitted graph and the initially selected detail. Hide
  interaction controls.
