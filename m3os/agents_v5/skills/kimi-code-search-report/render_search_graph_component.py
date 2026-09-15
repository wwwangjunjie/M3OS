#!/usr/bin/env python3
"""Render the reusable M3OS Molecular Search Graph HTML component."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


COMPONENT_VERSION = "m3os-molecular-search-graph-template-v2"
HUES = (217, 174, 38, 262, 345, 190)
DESCRIPTORS = (
    ("mw", "分子量", "Molecular weight"),
    ("clogp", "cLogP", "cLogP"),
    ("tpsa", "TPSA", "TPSA"),
    ("hba", "HBA", "HBA"),
    ("hbd", "HBD", "HBD"),
    ("rotatable_bonds", "可旋转键", "Rotatable bonds"),
    ("aromatic_rings", "芳香环", "Aromatic rings"),
    ("fraction_csp3", "Fsp3", "Fsp3"),
)


def esc(value: Any) -> str:
    return html.escape(str(value) if value is not None else "", quote=True)


def number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def strip_svg_dimensions(svg: str) -> str:
    text = str(svg or "")
    end = text.find(">")
    if end < 0:
        return text
    head = re.sub(r"\s(?:width|height)=(['\"]).*?\1", "", text[:end])
    return head + text[end:]


def language_text(language: str, zh: str, en: str) -> str:
    return zh if language.lower().startswith("zh") else en


def molecule_number(node: dict[str, Any]) -> int:
    try:
        value = int(node.get("molecule_number"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Graph node {node.get('id')!r} has no valid molecule_number"
        ) from exc
    if value < 1:
        raise ValueError(f"Graph node {node.get('id')!r} has an invalid molecule_number")
    return value


def molecule_label(node: dict[str, Any]) -> str:
    expected = f"Molecule {molecule_number(node)}"
    supplied = str(node.get("molecule_label") or expected)
    if supplied != expected:
        raise ValueError(
            f"Graph node {node.get('id')!r} has an inconsistent molecule_label"
        )
    return expected


def render_component(payload: dict[str, Any], assets: dict[str, Any]) -> str:
    graph = payload["search_graph"]
    semantics = payload.get("search_graph_semantics") or {}
    language = str((payload.get("report_request") or {}).get("language") or "en")
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
    if not nodes:
        raise ValueError("search_graph.nodes is empty")
    node_by_id = {str(node.get("id") or ""): node for node in nodes}
    if "" in node_by_id:
        raise ValueError("Every graph node must have a non-empty id")
    supplied_numbers = sorted(molecule_number(node) for node in nodes)
    if supplied_numbers != list(range(1, len(nodes) + 1)):
        raise ValueError(
            "search_graph.nodes must have unique, contiguous molecule numbers"
        )
    valid_ids = set(node_by_id)
    edges = [
        edge
        for edge in graph.get("edges", [])
        if isinstance(edge, dict)
        and str(edge.get("source") or "") in valid_ids
        and str(edge.get("target") or "") in valid_ids
    ]
    root = next((node for node in nodes if node.get("is_root")), nodes[0])
    best = next((node for node in nodes if node.get("is_best")), None)
    current = next((node for node in nodes if node.get("is_current")), None)
    initial = best or current or root
    candidates = assets.get("candidates") or {}
    root_asset = assets.get("root") or {}

    positions, depths, width, height, max_depth = compute_layout(
        nodes=nodes,
        edges=edges,
        root_id=str(root["id"]),
    )
    home_viewbox = f"0 0 {width} {height}"
    score_ranges: dict[int, tuple[float, float]] = {}
    for node in nodes:
        iteration = int(node.get("iteration") or 0)
        score = node.get("score")
        if isinstance(score, (int, float)):
            values = [
                float(item["score"])
                for item in nodes
                if int(item.get("iteration") or 0) == iteration
                and isinstance(item.get("score"), (int, float))
            ]
            score_ranges[iteration] = (min(values), max(values))

    def node_label(node: dict[str, Any]) -> str:
        return str(molecule_number(node))

    def node_colors(node: dict[str, Any]) -> tuple[str, str]:
        if node is root:
            return "#dbeafe", "#2563eb"
        if node.get("unreachable"):
            return "#cbd5e1", "#64748b"
        iteration = max(1, int(node.get("iteration") or 1))
        hue = HUES[(iteration - 1) % len(HUES)]
        low, high = score_ranges.get(iteration, (0.0, 0.0))
        score = node.get("score")
        ratio = 0.5
        if isinstance(score, (int, float)) and high > low:
            ratio = (float(score) - low) / (high - low)
        lightness = 80 - 32 * max(0.0, min(1.0, ratio))
        return f"hsl({hue},72%,{lightness:.0f}%)", f"hsl({hue},60%,34%)"

    svg: list[str] = [
        f'<svg data-graph-svg data-home-viewbox="{home_viewbox}" viewBox="{home_viewbox}" '
        'role="img" aria-label="Molecular Search Graph" preserveAspectRatio="xMidYMid meet">',
        '<defs><marker id="m3os-msg-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#94a3b8"/></marker></defs>',
        '<g data-graph-panzoom>',
    ]
    for depth in range(max_depth + 1):
        center_x = 120 + depth * 220
        label = "Root" if depth == 0 else language_text(language, f"第 {depth} 代", f"Layer {depth}")
        svg.append(
            f'<g data-graph-layer="{depth}"><rect x="{center_x - 62}" y="34" '
            f'width="268" height="{height - 68}" rx="18" fill="#f1f5f9" opacity="0.55"/>'
            f'<text x="{center_x + 72}" y="58" text-anchor="middle" font-size="14" '
            f'fill="#64748b" font-weight="600">{esc(label)}</text></g>'
        )
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        midpoint_x, midpoint_y = (x1 + x2) / 2, (y1 + y2) / 2
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
        action = str(edge.get("action") or edge.get("modification") or "")
        short_action = " ".join(action.split()[:3]) or language_text(language, "编辑", "edit")
        svg.append(
            f'<g data-graph-edge="{esc(source)}|{esc(target)}" data-source="{esc(source)}" '
            f'data-target="{esc(target)}"><line x1="{x1:.1f}" y1="{y1:.1f}" '
            f'x2="{x2:.1f}" y2="{y2:.1f}" stroke="#94a3b8" stroke-width="1.6" '
            'marker-end="url(#m3os-msg-arrow)"/>'
            f'<text x="{midpoint_x:.1f}" y="{midpoint_y - 6:.1f}" font-size="10" '
            f'fill="#64748b" text-anchor="middle" transform="rotate({angle:.1f} '
            f'{midpoint_x:.1f} {midpoint_y - 6:.1f})" pointer-events="none">'
            f'{esc(short_action)}<title>{esc(action)}</title></text></g>'
        )
    for node in sorted(
        nodes,
        key=lambda item: (
            depths[str(item["id"])],
            int(item.get("iteration") or 0),
            str(item["id"]),
        ),
    ):
        node_id = str(node["id"])
        x, y = positions[node_id]
        fill, stroke = node_colors(node)
        radius = 28 if node is root or node.get("is_best") or node.get("is_current") else 23
        if node.get("is_best") and node.get("is_current"):
            border = 'stroke="#1f2937" stroke-width="4"'
        elif node.get("is_best"):
            border = 'stroke="#10b981" stroke-width="3.5"'
        elif node.get("is_current"):
            border = 'stroke="#f59e0b" stroke-width="3.5"'
        else:
            border = f'stroke="{stroke}" stroke-width="2"'
        opacity = ' opacity="0.6"' if node.get("unreachable") else ""
        svg.append(
            f'<g data-graph-node="{esc(node_id)}" '
            f'data-molecule-number="{molecule_number(node)}" role="button" tabindex="0" '
            f'aria-pressed="false" aria-label="{esc(molecule_label(node))} {esc(node.get("smiles") or "")}" '
            f'transform="translate({x:.1f},{y:.1f})" class="m3os-msg-node"{opacity}>'
            f'<circle data-graph-node-hit="{esc(node_id)}" r="38" fill="#fff" '
            'fill-opacity="0" style="pointer-events:all"></circle>'
            f'<circle class="m3os-msg-node-visible" r="{radius}" fill="{fill}" {border} '
            'pointer-events="none"></circle>'
            f'<text y="4" text-anchor="middle" font-size="11" font-weight="700" '
            f'fill="#0f172a" pointer-events="none">{esc(node_label(node))}</text>'
            f'<text y="{radius + 16}" text-anchor="middle" font-size="11" fill="#334155" '
            f'pointer-events="none" class="m3os-msg-node-score">{esc(number(node.get("score")))}</text>'
            '</g>'
        )
    svg.append("</g></svg>")

    details = "".join(
        render_detail(
            node=node,
            root=root,
            initial=initial,
            node_by_id=node_by_id,
            root_asset=root_asset,
            candidate_assets=candidates,
            language=language,
        )
        for node in sorted(
            nodes,
            key=lambda item: (
                depths[str(item["id"])],
                int(item.get("iteration") or 0),
                str(item["id"]),
            ),
        )
    )
    best_score = best.get("score") if best else graph.get("summary", {}).get("best_score")
    round_count = semantics.get("optimization_round_count")
    if round_count is None:
        round_count = max((int(node.get("iteration") or 0) for node in nodes), default=0)
    title = language_text(language, "分子搜索演化图", "Molecular Search Graph")
    node_word = language_text(language, "节点", "Nodes")
    edge_word = language_text(language, "边", "Edges")
    best_word = language_text(language, "最佳分数", "Best score")
    round_word = language_text(language, "搜索轮数", "Rounds")
    current_label = language_text(language, "当前", "Current")
    layered_label = language_text(language, "按代际 / 父节点分层", "Layered by generation / parent")
    css = component_css()
    script = component_script(str(initial["id"]))
    return f'''<style data-m3os-graph-component-style="{COMPONENT_VERSION}">{css}</style>
<section class="m3os-msg" data-evolution-graph="m3os-molecular-search-graph-v1"
 data-graph-component-source="{COMPONENT_VERSION}">
  <div class="m3os-msg-heading"><div><span class="m3os-msg-eyebrow">M3OS</span>
    <h2>{esc(title)} <small>Molecular Search Graph</small></h2></div>
    <div data-graph-metrics class="m3os-msg-metrics">
      <span><label>{esc(node_word)}</label><b>{len(nodes)}</b></span>
      <span><label>{esc(edge_word)}</label><b>{len(edges)}</b></span>
      <span><label>{esc(best_word)}</label><b>{esc(number(best_score))}</b></span>
      <span><label>{esc(round_word)}</label><b>{esc(round_count)}</b></span>
    </div>
  </div>
  <div data-graph-inspector class="m3os-msg-inspector">
    <div data-graph-stage class="m3os-msg-stage">
      <div data-graph-controls class="m3os-msg-controls">
        <button type="button" data-graph-zoom="in" aria-label="Zoom in">+</button>
        <button type="button" data-graph-zoom="out" aria-label="Zoom out">−</button>
        <button type="button" data-graph-fit>Fit</button>
      </div>
      {''.join(svg)}
      <div data-graph-legend class="m3os-msg-legend">
        <span><i class="root"></i>Root</span><span><i class="round"></i>{esc(round_word)}</span>
        <span><i class="depth"></i>{esc(best_word)}</span><span><i class="best"></i>Best</span>
        <span><i class="current"></i>{esc(current_label)}</span><span><i class="layered"></i>{esc(layered_label)}</span>
      </div>
    </div>
    <div data-graph-node-details class="m3os-msg-details">{details}</div>
  </div>
</section>
<script data-m3os-graph-component-script="{COMPONENT_VERSION}">{script}</script>'''


def compute_layout(
    *, nodes: list[dict[str, Any]], edges: list[dict[str, Any]], root_id: str
) -> tuple[dict[str, tuple[float, float]], dict[str, int], int, int, int]:
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, list[str]] = defaultdict(list)
    node_by_id = {str(node["id"]): node for node in nodes}
    for edge in edges:
        source, target = str(edge["source"]), str(edge["target"])
        children[source].append(target)
        parents[target].append(source)
    depth = {root_id: 0}
    queue = deque([root_id])
    while queue:
        source = queue.popleft()
        for target in children.get(source, []):
            candidate_depth = depth[source] + 1
            if target not in depth or candidate_depth < depth[target]:
                depth[target] = candidate_depth
                queue.append(target)
    for node_id, node in node_by_id.items():
        depth.setdefault(node_id, 0 if node_id == root_id else max(1, int(node.get("iteration") or 1)))
    max_depth = max(depth.values(), default=0)
    left, top, layer_gap, column_gap, row_gap, group_gap = 120, 94, 220, 72, 88, 62

    def columns(count: int) -> int:
        return 1 if count <= 1 else (2 if count <= 4 else 3)

    groups_by_depth: dict[int, list[tuple[tuple[int, str | None], list[dict[str, Any]]]]] = {}
    for current_depth in range(1, max_depth + 1):
        grouped: dict[tuple[int, str | None], list[dict[str, Any]]] = defaultdict(list)
        for node in sorted(
            (item for item in nodes if depth[str(item["id"])] == current_depth),
            key=lambda item: (int(item.get("iteration") or 0), str(item["id"])),
        ):
            prior_parents = sorted(
                parent
                for parent in parents.get(str(node["id"]), [])
                if depth.get(parent) == current_depth - 1
            )
            grouped[(int(node.get("iteration") or current_depth), prior_parents[0] if prior_parents else None)].append(node)
        for members in grouped.values():
            members.sort(key=lambda item: (-(float(item.get("score") or -1)), str(item["id"])))
        groups_by_depth[current_depth] = list(grouped.items())

    def content_height(current_depth: int) -> int:
        heights = [
            (math.ceil(len(members) / columns(len(members))) - 1) * row_gap
            for _key, members in groups_by_depth.get(current_depth, [])
        ]
        return sum(heights) + group_gap * max(0, len(heights) - 1)

    content = max((content_height(item) for item in range(max_depth + 1)), default=0)
    height = max(560, content + top + 94)
    width = max(920, left + max_depth * layer_gap + 2 * column_gap + 120)
    if width / height < 1.64:
        width = round(height * 1.64)
    positions: dict[str, tuple[float, float]] = {root_id: (left, height / 2)}
    for current_depth in range(1, max_depth + 1):
        groups = sorted(
            groups_by_depth.get(current_depth, []),
            key=lambda item: (
                item[0][0],
                positions.get(item[0][1] or "", (0, 0))[1],
                str(item[0][1]),
            ),
        )
        group_heights = [
            (math.ceil(len(members) / columns(len(members))) - 1) * row_gap
            for _key, members in groups
        ]
        total = sum(group_heights) + group_gap * max(0, len(group_heights) - 1)
        cursor_y = (height - total) / 2
        for (_key, members), group_height in zip(groups, group_heights):
            group_columns = columns(len(members))
            for index, node in enumerate(members):
                positions[str(node["id"])] = (
                    left + current_depth * layer_gap + (index % group_columns) * column_gap,
                    cursor_y + (index // group_columns) * row_gap,
                )
            cursor_y += group_height + group_gap
    return positions, depth, width, height, max_depth


def render_detail(
    *,
    node: dict[str, Any],
    root: dict[str, Any],
    initial: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
    root_asset: dict[str, Any],
    candidate_assets: dict[str, Any],
    language: str,
) -> str:
    node_id = str(node["id"])
    active = " active" if node_id == str(initial["id"]) else ""
    is_root = node_id == str(root["id"])
    asset = root_asset if is_root else candidate_assets.get(node_id, {})
    topology = strip_svg_dimensions(str(asset.get("topology_svg") or asset.get("svg") or ""))
    descriptor_cells = "".join(
        f'<div><label>{esc(language_text(language, zh, en))}</label><b>{esc(number((asset.get("descriptors") or {}).get(key)))}</b></div>'
        for key, zh, en in DESCRIPTORS
        if (asset.get("descriptors") or {}).get(key) is not None
    )
    parent_ids = [str(value) for value in node.get("parent_ids", []) if str(value)]
    parent_smiles = [str(value) for value in node.get("parent_smiles", []) if str(value)]
    if not parent_smiles:
        parent_smiles = [str(node_by_id[value].get("smiles") or "") for value in parent_ids if value in node_by_id]
    parent_references = " | ".join(
        f'<span data-molecule-ref="{esc(parent_id)}" '
        f'data-molecule-number="{molecule_number(node_by_id[parent_id])}">'
        f'{esc(molecule_label(node_by_id[parent_id]))}</span>'
        for parent_id in parent_ids
        if parent_id in node_by_id
    )
    parent_block = ""
    if not is_root:
        parent_block = (
            f'<section><h4>{esc(language_text(language, "直接父节点", "Direct parent"))}</h4>'
            f'<div class="m3os-msg-parent-label">{parent_references or "N/A"}</div>'
            f'<div class="m3os-msg-smiles">{esc(" | ".join(parent_smiles) or "N/A")}</div></section>'
            f'<section><h4>{esc(language_text(language, "编辑操作", "Edit"))}</h4>'
            f'<p>{esc(node.get("action") or "N/A")}</p></section>'
        )
    rationale = "".join(
        f'<blockquote><b>{esc(label)}</b>{esc(text)}</blockquote>'
        for label, text in (
            (language_text(language, "生成器理由", "Generator rationale"), node.get("generator_rationale")),
            (language_text(language, "评论理由", "Critic rationale"), node.get("critic_rationale")),
        )
        if text
    )
    status = "Root" if is_root else ("Best" if node.get("is_best") else "")
    round_label = (
        f'{esc(language_text(language, "第", "Round "))} '
        f'{int(node.get("iteration") or 0)} '
        f'{esc(language_text(language, "轮", ""))}'
    )
    status_and_round = f"{esc(status)} · {round_label}" if status else round_label
    return (
        f'<article data-graph-node-detail="{esc(node_id)}" '
        f'data-molecule-number="{molecule_number(node)}" class="m3os-msg-detail{active}">'
        f'<header><h3>{esc(molecule_label(node))}</h3>'
        f'<span>{status_and_round}</span></header>'
        f'<div class="m3os-msg-smiles" data-full-smiles="{esc(node_id)}">{esc(node.get("smiles") or "")}</div>'
        f'<div class="m3os-msg-molecule">{topology}</div>{parent_block}{rationale}'
        f'<div class="m3os-msg-properties" data-property-grid="{esc(node_id)}-graph" '
        f'data-property-columns="2">{descriptor_cells}</div></article>'
    )


def component_css() -> str:
    return r'''
.m3os-msg{--msg-line:#dbe3ef;--msg-muted:#64748b;border:1px solid var(--msg-line);border-radius:14px;background:#f8fafc;padding:18px;min-width:0;color:#0f172a}
.m3os-msg *{box-sizing:border-box}.m3os-msg-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:12px}.m3os-msg-heading h2{margin:2px 0 0;font-size:clamp(20px,2vw,28px)}.m3os-msg-heading h2 small{font-size:.62em;color:var(--msg-muted)}.m3os-msg-eyebrow{font-size:11px;font-weight:800;letter-spacing:.16em;text-transform:uppercase;color:#2563eb}.m3os-msg-metrics{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:7px}.m3os-msg-metrics span{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--msg-line);background:#fff;border-radius:999px;padding:6px 10px}.m3os-msg-metrics label{font-size:11px;color:var(--msg-muted);font-weight:700}.m3os-msg-metrics b{font-size:13px}
.m3os-msg-inspector{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:16px;align-items:stretch}.m3os-msg-stage,.m3os-msg-details{min-width:0}.m3os-msg-stage{position:relative;display:grid;grid-template-rows:auto minmax(0,1fr) auto;border:1px solid var(--msg-line);border-radius:12px;background:#fff;overflow:hidden}.m3os-msg-controls{display:flex;gap:6px;padding:9px;border-bottom:1px solid var(--msg-line)}.m3os-msg-controls button{min-width:32px;height:30px;border:1px solid var(--msg-line);border-radius:8px;background:#fff;color:#1d4ed8;font-weight:800;cursor:pointer}.m3os-msg-controls button:hover{background:#eff6ff}
.m3os-msg [data-graph-svg]{display:block;width:100%;height:100%;min-height:390px;aspect-ratio:1.64/1;cursor:grab;touch-action:none}.m3os-msg [data-graph-svg]:active{cursor:grabbing}.m3os-msg-node{cursor:pointer;outline:none}.m3os-msg-node-visible{transition:stroke-width .15s ease,filter .15s ease}.m3os-msg-node:hover .m3os-msg-node-visible,.m3os-msg-node:focus-visible .m3os-msg-node-visible{stroke:#111827!important;stroke-width:3.5px!important;filter:drop-shadow(0 3px 7px rgba(15,23,42,.28))}.m3os-msg-node.selected .m3os-msg-node-visible{stroke:#111827!important;stroke-width:4px!important;filter:drop-shadow(0 4px 9px rgba(15,23,42,.32))}.m3os-msg-legend{display:flex;flex-wrap:wrap;gap:10px;padding:10px 12px;border-top:1px solid var(--msg-line);color:var(--msg-muted);font-size:11px;font-weight:700}.m3os-msg-legend span{display:inline-flex;align-items:center;gap:5px}.m3os-msg-legend i{width:9px;height:9px;border-radius:999px;background:#94a3b8}.m3os-msg-legend .root{background:#2563eb}.m3os-msg-legend .round{background:linear-gradient(135deg,#dbeafe,#2563eb)}.m3os-msg-legend .depth{background:linear-gradient(135deg,#bfdbfe,#1e3a8a)}.m3os-msg-legend .best{background:#10b981}.m3os-msg-legend .current{background:#f59e0b}.m3os-msg-legend .layered{background:repeating-linear-gradient(90deg,#dbeafe 0 4px,#bfdbfe 4px 8px)}
.m3os-msg-details{max-height:720px;overflow:auto;border:1px solid var(--msg-line);border-radius:12px;background:#fff;padding:14px}.m3os-msg-detail{display:none}.m3os-msg-detail.active{display:grid;gap:11px}.m3os-msg-detail header{display:flex;align-items:center;justify-content:space-between;gap:10px}.m3os-msg-detail h3,.m3os-msg-detail h4,.m3os-msg-detail p{margin:0}.m3os-msg-detail h4{font-size:12px;color:var(--msg-muted)}.m3os-msg-detail blockquote{margin:0;border-left:3px solid #93c5fd;background:#f8fafc;padding:9px 11px;font-size:12.5px;line-height:1.55}.m3os-msg-detail blockquote b{display:block;color:#334155;margin-bottom:3px}.m3os-msg-smiles{white-space:normal;overflow-wrap:anywhere;word-break:break-all;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;line-height:1.55}.m3os-msg-molecule{display:grid;place-items:center;min-height:190px}.m3os-msg-molecule svg{display:block;width:100%;height:auto;max-height:280px}.m3os-msg-properties{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.m3os-msg-properties>div{min-width:0;border:1px solid var(--msg-line);background:#f8fafc;border-radius:8px;padding:7px 9px}.m3os-msg-properties label{display:block;color:var(--msg-muted);font-size:10.5px}.m3os-msg-properties b{display:block;overflow-wrap:anywhere;font-size:13px;margin-top:2px}
@media(max-width:900px){.m3os-msg-heading{align-items:flex-start;flex-direction:column}.m3os-msg-metrics{justify-content:flex-start}.m3os-msg-inspector{grid-template-columns:minmax(0,1fr)}.m3os-msg-details{max-height:none}.m3os-msg [data-graph-svg]{min-height:340px}}
@media(max-width:560px){.m3os-msg{padding:11px}.m3os-msg-properties{grid-template-columns:minmax(0,1fr)}.m3os-msg [data-graph-svg]{min-height:300px}}
@media print{.m3os-msg-controls{display:none!important}.m3os-msg-inspector{grid-template-columns:1.5fr 1fr}.m3os-msg-details{max-height:none;overflow:visible}.m3os-msg-detail{display:none!important}.m3os-msg-detail.active{display:grid!important}}
'''


def component_script(initial_node_id: str) -> str:
    initial_json = json.dumps(initial_node_id, ensure_ascii=False)
    return r'''(function(){
"use strict";
var script=document.currentScript;
var section=script&&script.previousElementSibling;
if(!section||!section.matches('[data-evolution-graph="m3os-molecular-search-graph-v1"]'))return;
var svg=section.querySelector('[data-graph-svg]');
var detailsBox=section.querySelector('[data-graph-node-details]');
var raw=(svg.getAttribute('viewBox')||'0 0 920 560').split(/\s+/).map(Number);
var home=raw.slice(),vb=raw.slice(),nodes={},positions={},edges=[];
function apply(){svg.setAttribute('viewBox',vb.join(' '));}
function point(e){var r=svg.getBoundingClientRect();return{x:vb[0]+(e.clientX-r.left)/r.width*vb[2],y:vb[1]+(e.clientY-r.top)/r.height*vb[3]};}
function zoom(scale,px,py){var nw=Math.max(360,Math.min(1600,vb[2]*scale));var nh=Math.max(180,Math.min(900,vb[3]*scale));px=px==null?vb[0]+vb[2]/2:px;py=py==null?vb[1]+vb[3]/2:py;var fx=(px-vb[0])/vb[2],fy=(py-vb[1])/vb[3];vb=[px-nw*fx,py-nh*fy,nw,nh];apply();}
section.querySelectorAll('[data-graph-node]').forEach(function(g){var id=g.getAttribute('data-graph-node');nodes[id]=g;var m=/translate\(([-\d.]+)[ ,]+([-\d.]+)\)/.exec(g.getAttribute('transform')||'');positions[id]={x:Number(m&&m[1]),y:Number(m&&m[2])};});
section.querySelectorAll('[data-graph-edge]').forEach(function(g){edges.push({g:g,line:g.querySelector('line'),text:g.querySelector('text'),source:g.getAttribute('data-source'),target:g.getAttribute('data-target')});});
var selected='';
function select(id){if(!nodes[id])return;Object.keys(nodes).forEach(function(key){var active=key===id;nodes[key].classList.toggle('selected',active);nodes[key].setAttribute('aria-pressed',active?'true':'false');});section.querySelectorAll('[data-graph-node-detail]').forEach(function(item){item.classList.toggle('active',item.getAttribute('data-graph-node-detail')===id);});selected=id;detailsBox.scrollTop=0;}
function updateEdges(id){edges.forEach(function(edge){if(edge.source!==id&&edge.target!==id)return;var a=positions[edge.source],b=positions[edge.target];edge.line.setAttribute('x1',a.x);edge.line.setAttribute('y1',a.y);edge.line.setAttribute('x2',b.x);edge.line.setAttribute('y2',b.y);var mx=(a.x+b.x)/2,my=(a.y+b.y)/2,angle=Math.atan2(b.y-a.y,b.x-a.x)*180/Math.PI;if(angle>90)angle-=180;if(angle<-90)angle+=180;edge.text.setAttribute('x',mx);edge.text.setAttribute('y',my-6);edge.text.setAttribute('transform','rotate('+angle+' '+mx+' '+(my-6)+')');});}
Object.keys(nodes).forEach(function(id){var g=nodes[id],start=null,startClient=null,moved=false,pointerId=null,suppressClick=false;g.addEventListener('pointerdown',function(e){if(e.button!==0)return;start=point(e);startClient={x:e.clientX,y:e.clientY};moved=false;pointerId=e.pointerId;try{g.setPointerCapture(pointerId);}catch(_e){}e.stopPropagation();});g.addEventListener('pointermove',function(e){if(!start||e.pointerId!==pointerId)return;var distance=Math.hypot(e.clientX-startClient.x,e.clientY-startClient.y);if(!moved&&distance<5)return;moved=true;var next=point(e),dx=next.x-start.x,dy=next.y-start.y;positions[id].x+=dx;positions[id].y+=dy;start=next;g.setAttribute('transform','translate('+positions[id].x+','+positions[id].y+')');updateEdges(id);});function finish(e){if(e.pointerId!==pointerId)return;if(moved){suppressClick=true;}else{select(id);}start=null;startClient=null;pointerId=null;try{g.releasePointerCapture(e.pointerId);}catch(_e){}}g.addEventListener('pointerup',finish);g.addEventListener('pointercancel',finish);g.addEventListener('click',function(){if(suppressClick){suppressClick=false;return;}select(id);});g.addEventListener('keydown',function(e){if(e.key==='Enter'||e.key===' '){e.preventDefault();select(id);}});});
var pan=null;svg.addEventListener('pointerdown',function(e){if(e.target.closest('[data-graph-node]')||e.button!==0)return;pan={id:e.pointerId,x:e.clientX,y:e.clientY,vb:vb.slice()};try{svg.setPointerCapture(e.pointerId);}catch(_e){}});svg.addEventListener('pointermove',function(e){if(!pan||e.pointerId!==pan.id)return;var rect=svg.getBoundingClientRect();vb=[pan.vb[0]-(e.clientX-pan.x)*pan.vb[2]/rect.width,pan.vb[1]-(e.clientY-pan.y)*pan.vb[3]/rect.height,pan.vb[2],pan.vb[3]];apply();});function endPan(e){if(!pan||e.pointerId!==pan.id)return;var id=pan.id;pan=null;try{svg.releasePointerCapture(id);}catch(_e){}}svg.addEventListener('pointerup',endPan);svg.addEventListener('pointercancel',endPan);
section.querySelector('[data-graph-zoom="in"]').addEventListener('click',function(){zoom(.82);});section.querySelector('[data-graph-zoom="out"]').addEventListener('click',function(){zoom(1.18);});section.querySelector('[data-graph-fit]').addEventListener('click',function(){vb=home.slice();apply();});svg.addEventListener('wheel',function(e){e.preventDefault();var p=point(e);zoom(e.deltaY>0?1.08:.92,p.x,p.y);},{passive:false});
select(__INITIAL_NODE__);
})();'''.replace("__INITIAL_NODE__", initial_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    component = render_component(payload, assets)
    args.output.write_text(component + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size, "version": COMPONENT_VERSION}))


if __name__ == "__main__":
    main()
