#!/usr/bin/env python3

"""Render per-company DARP routes to HTML (SVG) and PNG."""

from __future__ import annotations

import html
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    return cleaned.strip("_") or "item"


def _normalize_coords(
    coords: Dict[str, List[float]],
    width: int,
    height: int,
    margin: int = 40,
) -> Dict[str, Tuple[float, float]]:
    xs = [c[0] for c in coords.values() if c is not None]
    ys = [c[1] for c in coords.values() if c is not None]
    if not xs or not ys:
        return {}
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x
    span_y = max_y - min_y
    span_x = span_x if span_x != 0 else 1.0
    span_y = span_y if span_y != 0 else 1.0

    scale_x = (width - 2 * margin) / span_x
    scale_y = (height - 2 * margin) / span_y

    norm: Dict[str, Tuple[float, float]] = {}
    for key, (x, y) in coords.items():
        nx = margin + (x - min_x) * scale_x
        ny = height - (margin + (y - min_y) * scale_y)
        norm[str(key)] = (nx, ny)
    return norm


def _node_label(node: Dict[str, Any]) -> str:
    actual = node.get("actual_location")
    entity_type = node.get("entity_type")
    entity_id = node.get("entity_id")
    location_type = node.get("location_type")
    if entity_type == "client":
        suffix = "PU" if location_type == "pickup" else "DO"
        return f"C{entity_id}_{suffix} (L{actual})"
    if entity_type == "vehicle":
        suffix = "S" if location_type == "start" else "E"
        return f"V{entity_id}_{suffix} (L{actual})"
    if actual is not None:
        return f"L{actual}"
    return "Node"


def _label_sort_key(node: Dict[str, Any]) -> Tuple[int, int]:
    location_type = node.get("location_type")
    entity_type = node.get("entity_type")
    priority_map = {
        "pickup": 0,
        "delivery": 1,
        "start": 2,
        "end": 3,
    }
    priority = priority_map.get(location_type, 9)
    entity_id = node.get("entity_id")
    if entity_id is None:
        entity_id = 99999
    if entity_type == "vehicle":
        priority += 5
    return (priority, int(entity_id) if isinstance(entity_id, int) else 99999)


def _collect_locations(route: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    locations: Dict[str, List[Dict[str, Any]]] = {}
    for node in route.get("node_sequence", []):
        actual = node.get("actual_location")
        if actual is None:
            continue
        key = str(actual)
        label_key = (
            node.get("entity_type"),
            node.get("entity_id"),
            node.get("location_type"),
        )
        if key not in locations:
            locations[key] = []
        # avoid duplicates
        if not any(
            (
                n.get("entity_type"),
                n.get("entity_id"),
                n.get("location_type"),
            ) == label_key
            for n in locations[key]
        ):
            locations[key].append(node)
    return locations


def _build_location_labels(nodes: List[Dict[str, Any]]) -> List[str]:
    ordered = sorted(nodes, key=_label_sort_key)
    labels = [_node_label(n) for n in ordered]
    # de-dup in case labels repeat
    seen: set[str] = set()
    uniq: List[str] = []
    for label in labels:
        if label not in seen:
            uniq.append(label)
            seen.add(label)
    return uniq


def _extract_route_edges(route: Dict[str, Any], time_matrix: List[List[float]]) -> List[Tuple[int, int, float]]:
    seq = route.get("node_sequence", [])
    edges: List[Tuple[int, int, float]] = []
    if len(seq) < 2:
        return edges
    for i in range(len(seq) - 1):
        a = seq[i]["node_index"]
        b = seq[i + 1]["node_index"]
        try:
            t = float(time_matrix[a][b])
        except Exception:
            t = float("nan")
        edges.append((a, b, t))
    return edges


def _svg_text(x: float, y: float, text: str, size: int = 12, color: str = "#222") -> str:
    safe = html.escape(text)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
        f'text-anchor="middle" dominant-baseline="central" '
        f'style="paint-order: stroke; stroke: #fff; stroke-width: 3">{safe}</text>'
    )


def _svg_multiline_text(x: float, y: float, lines: List[str], size: int = 12, color: str = "#222") -> str:
    if not lines:
        return ""
    tspans = []
    line_height = size + 2
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else line_height
        tspans.append(
            f'<tspan x="{x:.1f}" dy="{dy}">{html.escape(line)}</tspan>'
        )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
        f'text-anchor="middle" dominant-baseline="central" '
        f'style="paint-order: stroke; stroke: #fff; stroke-width: 3">{"".join(tspans)}</text>'
    )


def _render_html_map(
    case_name: str,
    company_name: str,
    route_summary: Dict[str, Any],
    data_model: Dict[str, Any],
    coordinates: Dict[str, List[float]],
    output_path: str,
) -> None:
    width = 1200
    height = 900
    margin = 50

    coords_norm = _normalize_coords(coordinates, width, height, margin)
    if not coords_norm:
        return

    time_matrix = data_model.get("time_matrix", [])
    routes = route_summary.get("routes", [])
    cmap = plt.get_cmap("tab10")

    svg_parts: List[str] = []

    # Draw all locations as light background points
    for loc_id, (x, y) in coords_norm.items():
        svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#c7c7c7" />')

    # Draw routes
    legend_entries: List[str] = []
    for idx, route in enumerate(routes):
        color = matplotlib_color(cmap(idx % 10))
        edges = _extract_route_edges(route, time_matrix)
        total_time = sum(t for _, _, t in edges if not math.isnan(t))
        vehicle_id = route.get("vehicle_id", idx)
        legend_entries.append(f"Vehicle {vehicle_id}: total_time={total_time:.1f}")

        seq = route.get("node_sequence", [])
        for i in range(len(seq) - 1):
            a = seq[i]
            b = seq[i + 1]
            a_loc = a.get("actual_location")
            b_loc = b.get("actual_location")
            if a_loc is None or b_loc is None:
                continue
            a_xy = coords_norm.get(str(a_loc))
            b_xy = coords_norm.get(str(b_loc))
            if not a_xy or not b_xy:
                continue
            svg_parts.append(
                f'<line x1="{a_xy[0]:.1f}" y1="{a_xy[1]:.1f}" '
                f'x2="{b_xy[0]:.1f}" y2="{b_xy[1]:.1f}" '
                f'stroke="{color}" stroke-width="2" />'
            )
            # Label edge with time
            edge_time = edges[i][2] if i < len(edges) else float("nan")
            label = f"t={edge_time:.1f}" if not math.isnan(edge_time) else "t=?"
            mid_x = (a_xy[0] + b_xy[0]) / 2
            mid_y = (a_xy[1] + b_xy[1]) / 2
            svg_parts.append(_svg_text(mid_x, mid_y, label, size=11))

        # Label nodes
        for loc_id, nodes in _collect_locations(route).items():
            xy = coords_norm.get(str(loc_id))
            if not xy:
                continue
            svg_parts.append(f'<circle cx="{xy[0]:.1f}" cy="{xy[1]:.1f}" r="6" fill="{color}" />')
            labels = _build_location_labels(nodes)
            svg_parts.append(_svg_multiline_text(xy[0], xy[1] - 14, labels, size=12))

    legend_html = "".join(f"<li>{html.escape(item)}</li>" for item in legend_entries)

    html_body = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>{html.escape(case_name)} | {html.escape(company_name)} route map</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
    .meta {{ margin-bottom: 12px; }}
    .legend {{ margin-top: 12px; }}
    svg {{ border: 1px solid #ccc; background: #fafafa; }}
  </style>
</head>
<body>
  <div class=\"meta\">
    <strong>Case:</strong> {html.escape(case_name)}<br />
    <strong>Company:</strong> {html.escape(company_name)}
  </div>
  <svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
    {"".join(svg_parts)}
  </svg>
  <div class=\"legend\">
    <strong>Route totals</strong>
    <ul>{legend_html}</ul>
  </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_body)


def matplotlib_color(color_tuple: Tuple[float, float, float, float]) -> str:
    r, g, b, _ = color_tuple
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _render_png_map(
    case_name: str,
    company_name: str,
    route_summary: Dict[str, Any],
    data_model: Dict[str, Any],
    coordinates: Dict[str, List[float]],
    output_path: str,
) -> None:
    if not coordinates:
        return

    time_matrix = data_model.get("time_matrix", [])
    routes = route_summary.get("routes", [])

    fig, ax = plt.subplots(figsize=(12, 9))
    coords_np = np.array(list(coordinates.values()), dtype=float)
    ax.scatter(coords_np[:, 0], coords_np[:, 1], s=30, color="#c7c7c7", zorder=1)

    cmap = plt.get_cmap("tab10")

    for idx, route in enumerate(routes):
        color = cmap(idx % 10)
        seq = route.get("node_sequence", [])
        edges = _extract_route_edges(route, time_matrix)
        total_time = sum(t for _, _, t in edges if not math.isnan(t))
        label = f"Vehicle {route.get('vehicle_id', idx)} total_time={total_time:.1f}"

        for i in range(len(seq) - 1):
            a = seq[i]
            b = seq[i + 1]
            a_loc = a.get("actual_location")
            b_loc = b.get("actual_location")
            if a_loc is None or b_loc is None:
                continue
            a_xy = coordinates.get(str(a_loc))
            b_xy = coordinates.get(str(b_loc))
            if not a_xy or not b_xy:
                continue
            ax.plot([a_xy[0], b_xy[0]], [a_xy[1], b_xy[1]], color=color, linewidth=2, zorder=2)
            edge_time = edges[i][2] if i < len(edges) else float("nan")
            mid_x = (a_xy[0] + b_xy[0]) / 2
            mid_y = (a_xy[1] + b_xy[1]) / 2
            edge_label = f"t={edge_time:.1f}" if not math.isnan(edge_time) else "t=?"
            ax.text(mid_x, mid_y, edge_label, fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

        for loc_id, nodes in _collect_locations(route).items():
            coord = coordinates.get(str(loc_id))
            if not coord:
                continue
            ax.scatter(coord[0], coord[1], s=80, color=color, zorder=3)
            labels = _build_location_labels(nodes)
            ax.text(coord[0], coord[1] + 0.2, "\n".join(labels), fontsize=8, ha="center")

        ax.plot([], [], color=color, label=label)

    ax.set_title(f"{case_name} | {company_name} routes")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    if routes:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_route_visuals(
    records: Iterable[Dict[str, Any]],
    output_dir: str,
    timestamp: str | None = None,
) -> str:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    visuals_dir = os.path.join(output_dir, "visuals", f"routes_{ts}")
    os.makedirs(visuals_dir, exist_ok=True)

    index_rows: List[str] = []

    for record in records:
        case_name = record["case_name"]
        company_name = record["company_name"]
        route_summary = record["route_summary"]
        data_model = record["data_model"]
        coordinates = record["coordinates"]

        base_name = f"{_slugify(case_name)}__{_slugify(company_name)}"
        html_path = os.path.join(visuals_dir, f"{base_name}.html")
        png_path = os.path.join(visuals_dir, f"{base_name}.png")

        _render_html_map(case_name, company_name, route_summary, data_model, coordinates, html_path)
        _render_png_map(case_name, company_name, route_summary, data_model, coordinates, png_path)

        index_rows.append(
            f"<tr><td>{html.escape(case_name)}</td><td>{html.escape(company_name)}</td>"
            f"<td><a href=\"{html.escape(os.path.basename(html_path))}\">HTML</a></td>"
            f"<td><a href=\"{html.escape(os.path.basename(png_path))}\">PNG</a></td></tr>"
        )

    index_html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Route Visuals</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; }}
    th {{ background: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>Route Visuals</h1>
  <table>
    <thead><tr><th>Case</th><th>Company</th><th>HTML</th><th>PNG</th></tr></thead>
    <tbody>
      {"".join(index_rows)}
    </tbody>
  </table>
</body>
</html>
"""

    index_path = os.path.join(visuals_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)

    return visuals_dir


def _build_case_svg(
    case_name: str,
    case_records: List[Dict[str, Any]],
    width: int = 1200,
    height: int = 900,
    margin: int = 50,
) -> str:
    # All records share same coordinates for a case
    coordinates = case_records[0]["coordinates"]
    coords_norm = _normalize_coords(coordinates, width, height, margin)
    if not coords_norm:
        return ""

    svg_parts: List[str] = []

    # Background points
    for loc_id, (x, y) in coords_norm.items():
        svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#c7c7c7" />')

    cmap = plt.get_cmap("tab10")

    for idx, record in enumerate(case_records):
        company_name = record["company_name"]
        route_summary = record["route_summary"]
        data_model = record["data_model"]
        time_matrix = data_model.get("time_matrix", [])

        color = matplotlib_color(cmap(idx % 10))
        group_id = f"{_slugify(case_name)}__{_slugify(company_name)}"
        svg_parts.append(f'<g class="company-group" data-company="{html.escape(company_name)}" id="{group_id}" style="display:none;">')

        # Draw each route (vehicle) in this company
        for r_idx, route in enumerate(route_summary.get("routes", [])):
            seq = route.get("node_sequence", [])
            for i in range(len(seq) - 1):
                a = seq[i]
                b = seq[i + 1]
                a_loc = a.get("actual_location")
                b_loc = b.get("actual_location")
                if a_loc is None or b_loc is None:
                    continue
                a_xy = coords_norm.get(str(a_loc))
                b_xy = coords_norm.get(str(b_loc))
                if not a_xy or not b_xy:
                    continue
                svg_parts.append(
                    f'<line x1="{a_xy[0]:.1f}" y1="{a_xy[1]:.1f}" '
                    f'x2="{b_xy[0]:.1f}" y2="{b_xy[1]:.1f}" '
                    f'stroke="{color}" stroke-width="2" />'
                )
                try:
                    edge_time = float(time_matrix[seq[i]["node_index"]][seq[i + 1]["node_index"]])
                except Exception:
                    edge_time = float("nan")
                label = f"t={edge_time:.1f}" if not math.isnan(edge_time) else "t=?"
                mid_x = (a_xy[0] + b_xy[0]) / 2
                mid_y = (a_xy[1] + b_xy[1]) / 2
                svg_parts.append(_svg_text(mid_x, mid_y, label, size=11))

            for loc_id, nodes in _collect_locations(route).items():
                xy = coords_norm.get(str(loc_id))
                if not xy:
                    continue
                svg_parts.append(f'<circle cx="{xy[0]:.1f}" cy="{xy[1]:.1f}" r="6" fill="{color}" />')
                labels = _build_location_labels(nodes)
                svg_parts.append(_svg_multiline_text(xy[0], xy[1] - 14, labels, size=12))

        svg_parts.append("</g>")

    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(svg_parts)}</svg>'


def render_routes_overview(
    records: Iterable[Dict[str, Any]],
    output_dir: str,
    timestamp: str | None = None,
) -> str:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    visuals_dir = os.path.join(output_dir, "visuals", f"routes_{ts}")
    os.makedirs(visuals_dir, exist_ok=True)

    case_map: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        case_map.setdefault(record["case_name"], []).append(record)

    case_names = sorted(case_map.keys())
    case_svgs: Dict[str, str] = {}
    case_companies: Dict[str, List[str]] = {}
    case_company_colors: Dict[str, Dict[str, str]] = {}

    for case_name in case_names:
        case_records = case_map[case_name]
        companies = [r["company_name"] for r in case_records]
        case_companies[case_name] = companies
        cmap = plt.get_cmap("tab10")
        case_company_colors[case_name] = {
            company: matplotlib_color(cmap(i % 10)) for i, company in enumerate(companies)
        }
        case_svgs[case_name] = _build_case_svg(case_name, case_records)

    options_html = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in case_names
    )
    checkbox_blocks = []
    legend_blocks = []
    for case_name in case_names:
        companies = case_companies[case_name]
        items = "".join(
            f'<label><input type="checkbox" class="company-check" data-case="{html.escape(case_name)}" '
            f'value="{html.escape(c)}" /> {html.escape(c)}</label>'
            for c in companies
        )
        checkbox_blocks.append(
            f'<div class="company-list" data-case="{html.escape(case_name)}" style="display:none;">{items}</div>'
        )
        legend_items = "".join(
            f'<span class="legend-item"><span class="swatch" style="background:{case_company_colors[case_name][c]};"></span>'
            f'{html.escape(c)}</span>'
            for c in companies
        )
        legend_blocks.append(
            f'<div class="legend" data-case="{html.escape(case_name)}" style="display:none;">'
            f'{legend_items}</div>'
        )

    svg_blocks = []
    for case_name in case_names:
        svg_blocks.append(
            f'<div class="case-svg" data-case="{html.escape(case_name)}" style="display:none;">'
            f'{case_svgs[case_name]}</div>'
        )

    overview_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Route Overview</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; color: #222; }}
    .controls {{ margin-bottom: 12px; }}
    .company-list label {{ display: inline-block; margin-right: 12px; }}
    .legend {{ margin: 10px 0 14px; }}
    .legend-item {{ display: inline-flex; align-items: center; margin-right: 12px; font-size: 13px; }}
    .swatch {{ width: 12px; height: 12px; display: inline-block; margin-right: 6px; border: 1px solid #333; }}
    svg {{ border: 1px solid #ccc; background: #fafafa; }}
  </style>
</head>
<body>
  <h1>Route Overview</h1>
  <div class="controls">
    <label>Case:
      <select id="caseSelect">{options_html}</select>
    </label>
  </div>
  {"".join(checkbox_blocks)}
  {"".join(legend_blocks)}
  {"".join(svg_blocks)}
  <script>
    const caseSelect = document.getElementById('caseSelect');
    const companyLists = document.querySelectorAll('.company-list');
    const caseSvgs = document.querySelectorAll('.case-svg');
    const legends = document.querySelectorAll('.legend');

    function updateCase() {{
      const current = caseSelect.value;
      companyLists.forEach(list => {{
        list.style.display = list.dataset.case === current ? 'block' : 'none';
        if (list.dataset.case !== current) {{
          list.querySelectorAll('input[type=checkbox]').forEach(cb => cb.checked = false);
        }}
      }});
      legends.forEach(legend => {{
        legend.style.display = legend.dataset.case === current ? 'block' : 'none';
      }});
      caseSvgs.forEach(svg => {{
        svg.style.display = svg.dataset.case === current ? 'block' : 'none';
        if (svg.dataset.case !== current) {{
          svg.querySelectorAll('.company-group').forEach(g => g.style.display = 'none');
        }}
      }});
    }}

    function updateCompanies() {{
      const current = caseSelect.value;
      const currentSvg = document.querySelector('.case-svg[data-case=\"' + current + '\"]');
      if (!currentSvg) return;
      const checks = document.querySelectorAll('.company-check[data-case=\"' + current + '\"]');
      const selected = new Set();
      checks.forEach(cb => {{ if (cb.checked) selected.add(cb.value); }});
      currentSvg.querySelectorAll('.company-group').forEach(group => {{
        group.style.display = selected.has(group.dataset.company) ? 'block' : 'none';
      }});
    }}

    caseSelect.addEventListener('change', () => {{
      updateCase();
      updateCompanies();
    }});
    document.querySelectorAll('.company-check').forEach(cb => {{
      cb.addEventListener('change', updateCompanies);
    }});

    updateCase();
  </script>
</body>
</html>
"""

    overview_path = os.path.join(visuals_dir, "overview.html")
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write(overview_html)

    return overview_path


__all__ = ["render_route_visuals", "render_routes_overview"]
