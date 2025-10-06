import os
import time
import json
import datetime
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import networkx as nx
import plotly.graph_objects as go

from flask import current_app
from dash.exceptions import PreventUpdate
from plotly.colors import sample_colorscale
from dash import Output, Input, State, ctx, html

# Optional Markdown (fallback to plain text if the package is unavailable)
try:
    import markdown  # type: ignore

    def md_to_html(text: str) -> str:
        return markdown.markdown(text)

except Exception:

    def md_to_html(text: str) -> str:
        return text


from config import GRAPH_DATA_PATH, GRAPH_FILE_PATTERN
from data_loader.loader import (
    get_graph_collections,
    find_graph_files,
    load_graph_from_path,
    load_graph_path,
)

# --------------------------------------------
# Caching helpers
# --------------------------------------------


def get_app_cache_dict(name: str) -> Dict[str, Any]:
    """
    Ensure a dict-like cache exists on the Flask current_app under the given name.
    Returns the dict for further use.
    """
    if not hasattr(current_app, name):
        setattr(current_app, name, {})
    return getattr(current_app, name)


def _layout_cache_key(graph: nx.Graph, layout_method: str) -> str:
    """
    Build a cache key for a given graph and layout method.
    """
    return f"{id(graph)}_{layout_method}_{len(list(graph.nodes()))}"


def _update_node_position_from_drag(point: dict) -> bool:
    """
    Zapisuje nową pozycję węzła, jeśli punkt zawiera:
    - customdata – id węzła (lub lista z id)
    - x, y       – współrzędne w przestrzeni wykresu

    Zwraca True, gdy pozycja została zmieniona, w przeciwnym razie False.
    """
    if not isinstance(point, dict):
        return False
    if "customdata" not in point or "x" not in point or "y" not in point:
        return False

    node_id = point["customdata"]
    if isinstance(node_id, list) and node_id:
        node_id = node_id[0]

    try:
        x_new = float(point["x"])
        y_new = float(point["y"])
    except (TypeError, ValueError):
        return False

    if not isinstance(getattr(current_app, "node_positions", {}), dict):
        current_app.node_positions = {}

    if current_app.node_positions.get(node_id) == (x_new, y_new):
        return False

    current_app.node_positions[node_id] = (x_new, y_new)
    return True


# --------------------------------------------
# Layout computation and trace builders
# --------------------------------------------


def compute_layout_optimized(
    graph: nx.Graph,
    layout_method: str = "spring",
    force_recompute: bool = False,
) -> Dict[Any, Tuple[float, float]]:
    """
    Compute and cache a 2D layout for the given graph.
    Uses different parameters for small vs large graphs to balance quality/performance.
    """
    layout_cache = get_app_cache_dict("layout_cache")
    cache_key = _layout_cache_key(graph, layout_method)

    if not force_recompute and cache_key in layout_cache:
        # Keep user-dragged positions if present
        cached_pos = layout_cache[cache_key]
        if (
            isinstance(getattr(current_app, "node_positions", {}), dict)
            and current_app.node_positions
        ):
            cached_pos = cached_pos.copy()
            cached_pos.update(current_app.node_positions)
        return cached_pos

    # Parameter presets tuned for performance on large graphs
    if len(list(graph.nodes())) > 1000:
        if layout_method == "spring":
            pos = nx.spring_layout(graph, k=0.5, iterations=20, seed=42)
        elif layout_method == "circular":
            pos = nx.circular_layout(graph)
        elif layout_method == "random":
            pos = nx.random_layout(graph, seed=42)
        elif layout_method == "shell":
            pos = nx.spring_layout(graph, k=1, iterations=10, seed=42)
        else:
            pos = nx.spring_layout(graph, k=0.5, iterations=20, seed=42)
    else:
        if layout_method == "spring":
            pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
        elif layout_method == "circular":
            pos = nx.circular_layout(graph)
        elif layout_method == "random":
            pos = nx.random_layout(graph, seed=42)
        elif layout_method == "shell":
            pos = nx.shell_layout(graph)
        else:
            pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)

    layout_cache[cache_key] = pos
    current_app.node_positions = pos.copy()
    return pos


def prepare_edge_traces(
    graph: nx.Graph,
    pos: Dict[Any, Tuple[float, float]],
    selected_nodes: Optional[List[Any]] = None,
    distance_info: Optional[Dict[Any, int]] = None,
) -> List[go.Scatter]:
    """
    Build Plotly edge traces grouped by styling categories:
    - normal: default, semi-transparent
    - selected: edges between selected nodes
    - distance: edges between nodes in the distance set
    """
    traces: List[go.Scatter] = []
    num_edges = len(graph.edges())

    selected_set = set(selected_nodes) if selected_nodes else set()
    distance_nodes = set(distance_info.keys()) if distance_info else set()

    edge_types: Dict[str, Dict[str, Any]] = {
        "normal": {
            "x": [],
            "y": [],
            "color": "rgba(125, 125, 125, 0.2)",
            "width": 0.5,
        },
        "selected": {"x": [], "y": [], "color": "rgba(255, 0, 0, 0.8)", "width": 2},
        "distance": {
            "x": [],
            "y": [],
            "color": "rgba(0, 150, 255, 0.6)",
            "width": 1.5,
        },
    }

    edges_to_draw = list(graph.edges(data=True))
    if num_edges > 5000:
        # Prioritize by weight if available
        edges_with_weights = [
            (u, v, d.get("weight", 1), d) for u, v, d in edges_to_draw
        ]
        edges_with_weights.sort(key=lambda x: x[2], reverse=True)
        edges_to_draw = [(u, v, d) for u, v, _, d in edges_with_weights[:5000]]

    for u, v, data in edges_to_draw:
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            coords_x = [x0, x1, None]
            coords_y = [y0, y1, None]

            if u in selected_set and v in selected_set:
                edge_type = "selected"
            elif u in distance_nodes and v in distance_nodes:
                edge_type = "distance"
            else:
                edge_type = "normal"

            edge_types[edge_type]["x"].extend(coords_x)
            edge_types[edge_type]["y"].extend(coords_y)

    for edge_type, data in edge_types.items():
        if data["x"]:
            traces.append(
                go.Scatter(
                    x=data["x"],
                    y=data["y"],
                    mode="lines",
                    line=dict(width=data["width"], color=data["color"]),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{edge_type.title()} Edges",
                )
            )

    return traces


def prepare_node_traces_with_distance(
    graph: nx.Graph,
    pos: Dict[Any, Tuple[float, float]],
    show_labels: bool,
    node_size_range: Tuple[int, int],
    selected_nodes: Optional[List[Any]] = None,
    distance_info: Optional[Dict[Any, int]] = None,
    max_nodes_for_labels: int = 5000,
    max_nodes_for_hover: int = 5000,
) -> List[go.Scatter]:
    """
    Build Plotly node traces grouped by categories
    (normal, selected, and distance bands).
    """
    traces: List[go.Scatter] = []
    num_nodes = len(list(graph.nodes()))

    selected_set = set(selected_nodes) if selected_nodes else set()

    node_categories: Dict[str, Dict[str, Any]] = {
        "normal": {"nodes": [], "color": "blue", "name": "Normal Nodes"},
        "selected": {"nodes": [], "color": "red", "name": "Selected Nodes"},
        "distance_1": {"nodes": [], "color": "orange", "name": "Distance 1"},
        "distance_2_3": {"nodes": [], "color": "gold", "name": "Distance 2-3"},
        "distance_4_6": {"nodes": [], "color": "lightgreen", "name": "Distance 4-6"},
        "distance_7_10": {
            "nodes": [],
            "color": "lightblue",
            "name": "Distance 7-10",
        },
        "distance_11_50": {"nodes": [], "color": "white", "name": "Distance 11-50"},
    }

    for node in graph.nodes():
        if node not in pos:
            continue

        if node in selected_set:
            category = "selected"
        elif distance_info and node in distance_info:
            distance = distance_info[node]
            if distance == 1:
                category = "distance_1"
            elif distance <= 3:
                category = "distance_2_3"
            elif distance <= 6:
                category = "distance_4_6"
            elif distance <= 10:
                category = "distance_7_10"
            elif distance <= 50:
                category = "distance_11_50"
            else:
                category = "normal"
        else:
            category = "normal"

        node_categories[category]["nodes"].append(node)

    all_weights = [graph.nodes[n].get("weight", 1) for n in graph.nodes()]
    min_weight, max_weight = min(all_weights), max(all_weights)

    traces_out: List[go.Scatter] = []
    for category, data in node_categories.items():
        if not data["nodes"]:
            continue

        nodes = data["nodes"]
        x_coords = [pos[node][0] for node in nodes]
        y_coords = [pos[node][1] for node in nodes]

        sizes: List[float] = []
        colors: List[float] = []
        nodes_labels: List[str] = []
        hovers: List[str] = []
        article_max_lin_len = 80

        for node in nodes:
            if not graph.has_node(node):
                continue
            node_data = graph.nodes[node].get("data", {})
            date = node_data.get("date", "unknown")
            date_week_day = "unknown"
            if date != "unknown":
                try:
                    date_date = datetime.datetime.strptime(date, "%Y.%m.%d")
                    date_week_day = date_date.strftime("%A")
                except Exception:
                    date_week_day = "unknown"

            size_val = node_data.get("size", "unknown")
            text = node_data.get("text", "unknown")
            if text != "unknown":
                text = md_to_html(text)
                new_text_merged = ""
                first_line = True
                text_lines = text.split("\n")
                for html_line in text_lines:
                    html_line = html_line.replace("<h1>", "<b>").replace(
                        "</h1>", "</b>"
                    )
                    html_line = html_line.replace("<h2>", "<b>").replace(
                        "</h2>", "</b>"
                    )
                    html_line = html_line.replace("<h3>", "<b>").replace(
                        "</h3>", "</b>"
                    )
                    html_line = html_line.replace("<h4>", "<b>").replace(
                        "</h4>", "</b>"
                    )
                    html_line = html_line.replace("</p>", "<br>").replace("<p>", "")
                    html_line = html_line.replace("<strong>", "<b>").replace(
                        "</strong>", "</b>"
                    )
                    html_line = html_line.replace("<li>", "* ").replace(
                        "</li>", "<br>"
                    )
                    html_line = html_line.replace("<ul>", "").replace("</ul>", "")
                    html_line = html_line.replace("%", "percent")

                    proper_line = ""
                    for _line_i in range(0, len(html_line), article_max_lin_len):
                        proper_line += html_line[
                            _line_i : _line_i + article_max_lin_len
                        ].strip()
                        proper_line += "<br>"

                    if first_line:
                        proper_line += "</b>"
                        first_line = True
                    new_text_merged += proper_line

                if len(new_text_merged.strip()):
                    new_text_merged = new_text_merged.replace(
                        "<br><br><br>", "<br><br>"
                    )
                    text = new_text_merged

            if isinstance(text, str) and len(text) > 2000:
                text = text[:2000] + " (...)"

            node_label = node_data.get("label", str(node))
            weight = float(graph.nodes[node].get("weight", size_val))

            if min_weight == max_weight:
                size = node_size_range[0]
            else:
                normalized = (weight - min_weight) / (max_weight - min_weight)
                size_range = node_size_range[1] - node_size_range[0]
                size = node_size_range[0] + size_range * normalized

            if category == "selected":
                size *= 1.5
            elif category.startswith("distance_"):
                size *= 1.2

            if date != "unknown":
                node_label = "<b>" + date + "</b> <br>" + node_label

            sizes.append(size)
            colors.append(weight)
            nodes_labels.append(node_label)

            hover_info = f"<b>Weekday</b>: {date_week_day} ({date})<br><br>"
            hover_info += text if isinstance(text, str) else ""
            hovers.append(hover_info)

        trace = go.Scatter(
            x=x_coords,
            y=y_coords,
            mode=(
                "markers+text"
                if show_labels and num_nodes <= max_nodes_for_labels
                else "markers"
            ),
            marker=dict(
                size=sizes,
                color=data["color"] if category != "normal" else colors,
                colorscale="Viridis" if category == "normal" else None,
                colorbar=(
                    dict(
                        title="Size of node",
                        titleside="right",
                        thickness=15,
                        len=0.7,
                    )
                    if category == "normal" and num_nodes <= 2000
                    else None
                ),
                line=dict(
                    width=3 if category == "selected" else 1,
                    color="darkred" if category == "selected" else "white",
                ),
                opacity=1.0 if category in ["selected", "distance_1"] else 0.8,
                sizemode="diameter",
            ),
            text=(
                nodes_labels
                if show_labels and num_nodes <= max_nodes_for_labels
                else None
            ),
            textposition="middle center",
            textfont=dict(
                size=(
                    16 if category == "selected" else 14 if num_nodes <= 500 else 12
                ),
                color="black",
            ),
            hovertext=hovers,
            hoverinfo=("text" if num_nodes <= max_nodes_for_hover else "text+name"),
            showlegend=True if distance_info else False,
            name=data["name"],
            customdata=nodes,
            legendgroup=category,
        )
        traces_out.append(trace)

    return traces_out


def prepare_interactive_traces(
    graph: nx.Graph,
    layout_method: str = "spring",
    show_labels: bool = True,
    node_size_range: Tuple[int, int] = (50, 75),
    selected_nodes: Optional[List[Any]] = None,
    distance_info: Optional[Dict[Any, int]] = None,
) -> List[go.Scatter]:
    """
    Prepare both edge and node traces for the interactive figure.
    """
    if not graph:
        return []

    pos = getattr(current_app, "node_positions", {}) or {}
    if not pos:
        pos = compute_layout_optimized(graph, layout_method)
    else:
        # Sync layout cache with current positions, but do not trigger extra updates
        cache_key = _layout_cache_key(graph, layout_method)
        layout_cache = get_app_cache_dict("layout_cache")
        layout_cache[cache_key] = pos

    traces: List[go.Scatter] = []
    traces.extend(prepare_edge_traces(graph, pos, selected_nodes, distance_info))
    traces.extend(
        prepare_node_traces_with_distance(
            graph,
            pos,
            show_labels,
            node_size_range,
            selected_nodes,
            distance_info,
            max_nodes_for_labels=getattr(current_app, "max_nodes_for_labels", 5000),
            max_nodes_for_hover=getattr(current_app, "max_nodes_for_hover", 5000),
        )
    )
    return traces


def create_interactive_figure(
    graph: nx.Graph,
    title: str,
    layout_method: str = "spring",
    show_labels: bool = True,
    node_size_range: Tuple[int, int] = (50, 75),
    selected_nodes: Optional[List[Any]] = None,
    distance_info: Optional[Dict[Any, int]] = None,
) -> go.Figure:
    """
    Create a Plotly figure displaying the graph with interactive capabilities.
    """
    traces = prepare_interactive_traces(
        graph,
        layout_method,
        show_labels,
        node_size_range,
        selected_nodes,
        distance_info,
    )

    fig = go.Figure(data=traces)
    num_nodes = len(list(graph.nodes())) if graph else 0
    show_legend = bool(distance_info)

    fig.update_layout(
        title=dict(
            text=(f"{title} ({num_nodes:,} nodes)" if graph else title),
            x=0.5,
            font=dict(size=18),
        ),
        showlegend=show_legend,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(
            showgrid=True,
            zeroline=False,
            showticklabels=True,
            gridwidth=1,
            gridcolor="rgba(235, 235, 235, 0.3)",
            fixedrange=False,
            title="X Coordinate",
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showticklabels=True,
            gridwidth=1,
            gridcolor="rgba(235, 235, 235, 0.3)",
            fixedrange=False,
            title="Y Coordinate",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=800,
        dragmode="pan",
        clickmode="event+select",
        transition={"duration": 0 if num_nodes > 1000 else 300},
    )

    if show_legend:
        fig.update_layout(
            legend=dict(
                x=1.02,
                y=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
            )
        )

    if graph:
        instructions = (
            "Nodes: {:,} | Click a node to explore "
            "neighborhood | Max distance: up to 10 hops"
        ).format(num_nodes)
        if distance_info:
            instructions += f" | Found: {len(distance_info)} nodes in range"

        fig.add_annotation(
            text=instructions,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.005,
            y=0.995,
            xanchor="left",
            yanchor="top",
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1,
        )

    return fig


# --------------------------------------------
# Distance exploration
# --------------------------------------------


def get_nodes_within_distance(
    graph: nx.Graph, start_node: Any, max_distance: int = 50
) -> Dict[Any, int]:
    """
    Perform a BFS from start_node up to max_distance hops and return a dict with
    mapping node -> distance. Results are cached per (graph, start_node, max_distance).
    """
    if not graph or start_node not in graph.nodes():
        return {}

    distance_cache = get_app_cache_dict("distance_cache")
    cache_key = f"{start_node}_{max_distance}_{id(graph)}"
    if cache_key in distance_cache:
        return distance_cache[cache_key]

    visited: Dict[Any, int] = {start_node: 0}
    queue: deque[Tuple[Any, int]] = deque([(start_node, 0)])

    while queue:
        current_node, current_distance = queue.popleft()

        if current_distance >= max_distance:
            continue

        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                new_distance = current_distance + 1
                visited[neighbor] = new_distance
                queue.append((neighbor, new_distance))

    distance_cache[cache_key] = visited
    return visited


# --------------------------------------------
# Dash callbacks registration
# --------------------------------------------


def register_callbacks(app) -> None:
    """
    Register all Dash callbacks for the interactive graph explorer.
    Initializes simple in-process caches on current_app.
    """
    # Defaults
    current_app.max_nodes_for_labels = 5000
    current_app.max_nodes_for_hover = 5000
    if not hasattr(current_app, "graph_cache"):
        current_app.graph_cache = {}
    if not hasattr(current_app, "layout_cache"):
        current_app.layout_cache = {}
    if not hasattr(current_app, "node_positions"):
        current_app.node_positions = {}
    if not hasattr(current_app, "distance_cache"):
        current_app.distance_cache = {}

    # 1) Scan collections on initial render (triggered once via component id)
    @app.callback(
        Output("collection-selector", "options"),
        Input("collection-selector", "id"),
    )
    def update_collection_dropdown(_):
        collections = get_graph_collections(GRAPH_DATA_PATH)
        if not collections:
            return [
                {
                    "label": "No collections found in 'data' directory",
                    "value": "",
                    "disabled": True,
                }
            ]
        return [{"label": col, "value": col} for col in collections]

    # 2) After selecting a collection, list graph files and enable the dropdown
    @app.callback(
        Output("graph-selector", "options"),
        Output("graph-selector", "value"),
        Output("graph-selector", "disabled"),
        Output("graph-paths-store", "data"),
        Input("collection-selector", "value"),
        prevent_initial_call=True,
    )
    def update_graph_dropdown(selected_collection):
        if not selected_collection:
            return [], None, True, None

        collection_path = os.path.join(GRAPH_DATA_PATH, selected_collection)
        graph_files = find_graph_files(str(collection_path), GRAPH_FILE_PATTERN)

        if not graph_files:
            return (
                [{"label": "No graphs found", "value": "", "disabled": True}],
                None,
                False,
                None,
            )

        options = []
        # Best-effort: append approximate file size without loading the graph
        for i, p in enumerate(graph_files):
            label = os.path.basename(p)
            options.append({"label": label, "value": i})
        # Sort by label
        options.sort(key=lambda o: o["label"])

        return options, None, False, graph_files

    # 3) Load the selected graph (into cache), update status and store meta
    @app.callback(
        Output("graph-load-status", "children"),
        Output("graph-load-status", "style"),
        Output("graph-store", "data"),
        Input("graph-selector", "value"),
        State("graph-paths-store", "data"),
        prevent_initial_call=True,
    )
    def load_selected_graph(selected_index, graph_paths):
        if selected_index is None or not graph_paths:
            raise PreventUpdate

        try:
            selected_path = graph_paths[selected_index]
            start_time = time.time()

            graph = load_graph_from_path(selected_path, current_app.graph_cache)

            load_time = time.time() - start_time

            if graph:
                num_nodes = len(graph.nodes())
                num_edges = len(graph.edges())
                is_multigraph = isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph))

                # Reset per-graph caches
                current_app.node_positions = {}
                current_app.distance_cache = {}
                # layout_cache is kept unless explicitly refreshed

                status = (
                    f"{os.path.basename(selected_path)} ({load_time:.1f}s) "
                    f"💥 Press 'Load Graph' to load visualisation"
                )
                style = {"fontSize": "12px", "color": "#27ae60"}
                graph_data = {
                    "path": selected_path,
                    "nodes": num_nodes,
                    "edges": num_edges,
                    "is_multigraph": is_multigraph,
                    "title": "Information Graph Explorer",
                }
            else:
                status = f"✗ Failed to load: {os.path.basename(selected_path)}"
                style = {"fontSize": "12px", "color": "#e74c3c"}
                graph_data = None

            return status, style, graph_data

        except Exception as e:
            return (
                f"✗ Error: {str(e)}",
                {"fontSize": "12px", "color": "#e74c3c"},
                None,
            )

    @app.callback(
        Output("drag-mode-store", "data"),
        Output("network-graph", "style"),
        Output("cytoscape-graph", "style"),
        Output("cytoscape-legend", "style"),
        Input("toggle-drag-btn", "n_clicks"),
        State("drag-mode-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_drag_mode(n_clicks, mode_data):
        """
        Toggles between Plotly (no drag) and Cytoscape (with drag).
        Also hides the Cytoscape legend when in Plotly mode.
        """
        current_mode = mode_data.get("mode", "plotly")
        new_mode = "cytoscape" if current_mode == "plotly" else "plotly"

        if new_mode == "cytoscape":
            # Show Cytoscape graph and its legend
            return (
                {"mode": "cytoscape"},
                {"display": "none"},
                {"width": "100%", "height": "800px", "display": "block"},
                {"display": "block"},
            )
        else:
            # Hide Cytoscape graph and its legend, show Plotly graph
            return (
                {"mode": "plotly"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
            )

    # NEW: Update Cytoscape graph
    # ile przedziałów (binów) chcesz pokazać w legendzie?
    COLOR_BINS = 5
    COLOR_SCALE = (
        "Viridis"  # dowolny z plotly.colors.sequential, np. "Turbo", "Plasma"
    )

    @app.callback(
        Output("cytoscape-graph", "elements"),
        Input("graph-store", "data"),
        Input("layout-dropdown", "value"),
        Input("exploration-state", "data"),
        Input("node-size-slider", "value"),
        State("drag-mode-store", "data"),
        prevent_initial_call=True,
    )
    def update_cytoscape_graph(
        graph_data, layout_method, exploration_state, node_size_range, mode_data
    ):
        """Generuje elementy Cytoscape – kolor i rozmiar wyliczane z wagi w `node["data"]["weight"]`."""
        if mode_data.get("mode") != "cytoscape" or not graph_data:
            raise PreventUpdate

        # ---------- wczytanie grafu ----------
        graph_path = graph_data.get("path")
        if not graph_path:
            raise PreventUpdate

        graph = load_graph_path(graph_path, current_app.graph_cache)
        if graph is None:
            return []

        # ---------- layout ----------
        pos = getattr(current_app, "node_positions", {}) or {}
        if not pos:
            pos = compute_layout_optimized(graph, layout_method)

        # ---------- pomocnicze dane ----------
        center_node = exploration_state.get("center_node")
        distance_info = exploration_state.get("distance_info", {})

        # wszystkie wagi – pobierane z node_data["weight"]
        all_weights = [
            graph.nodes[n].get("data", {}).get("weight", 10) for n in graph.nodes()
        ]
        min_w, max_w = min(all_weights), max(all_weights)

        # podział na przedziały – potrzebny do legendy
        bin_edges = np.linspace(min_w, max_w, COLOR_BINS + 1)

        # budujemy mapę kolor‑>etykieta (ta sama dla całego grafu)
        legend_map = {}
        for i in range(COLOR_BINS):
            # środek przedziału
            low, high = bin_edges[i], bin_edges[i + 1]
            mid = (low + high) / 2
            norm_mid = (mid - min_w) / (max_w - min_w) if max_w != min_w else 0.5
            col = sample_colorscale(COLOR_SCALE, [norm_mid])[0]  # np. "rgb(68,1,84)"
            label = f"{low:.1f} – {high:.1f}"
            legend_map[col] = label

        # zapamiętujemy mapę legendy (przydatne w drugim callbacku)
        current_app._cyto_legend_map = legend_map

        elements = []

        for node in graph.nodes():
            if node not in pos:
                continue

            x, y = pos[node]
            node_data = graph.nodes[node].get("data", {})
            label = node_data.get("label", str(node))
            weight = node_data.get("weight", 10)

            # ---------- rozmiar (tak jak w Plotly) ----------
            if min_w == max_w:
                size = node_size_range[0]
            else:
                norm = (weight - min_w) / (max_w - min_w)
                size_range = node_size_range[1] - node_size_range[0]
                size = node_size_range[0] + size_range * norm

            # ---------- scaling wybranego węzła ----------
            size_selected = size
            if node == center_node:
                size_selected = size * 1.5
            elif distance_info and node in distance_info:
                size_selected = size * 1.2

            # ---------- kolor z wagi ----------
            norm_weight = (
                0.5 if max_w == min_w else (weight - min_w) / (max_w - min_w)
            )
            color = sample_colorscale(COLOR_SCALE, [norm_weight])[0]

            # ---------- klasy CSS ----------
            classes = []
            if node == center_node:
                classes.append("selected")
            elif distance_info and node in distance_info:
                classes.append("distance")

            # ---------- element węzła ----------
            elements.append(
                {
                    "data": {
                        "id": str(node),
                        "label": label,
                        "size": size,
                        "size_selected": size_selected,
                        "color": color,  # Cytoscape użyje tego w stylesheet
                        "color_label": legend_map[
                            color
                        ],  # etykieta, przydatna w legendzie (opcjonalnie)
                    },
                    "position": {"x": x * 500, "y": y * 500},
                    "classes": " ".join(classes),
                }
            )

        # ---------- krawędzie ----------
        for u, v in list(graph.edges())[:2000]:
            if u in pos and v in pos:
                elements.append({"data": {"source": str(u), "target": str(v)}})

        return elements

    # NEW: Handle Cytoscape node clicks
    @app.callback(
        Output("exploration-state", "data", allow_duplicate=True),
        Input("cytoscape-graph", "tapNodeData"),
        Input("distance-slider", "value"),
        State("exploration-state", "data"),
        State("graph-store", "data"),
        prevent_initial_call=True,
    )
    def handle_cytoscape_click(
        tap_node_data, max_distance, exploration_state, graph_data
    ):
        """
        Handles node clicks in Cytoscape mode.
        """
        if not tap_node_data or not graph_data:
            raise PreventUpdate

        graph_path = graph_data.get("path")
        graph = load_graph_from_path(graph_path, current_app.graph_cache)
        if graph is None:
            raise PreventUpdate

        clicked_node = tap_node_data.get("id")
        center_node = exploration_state.get("center_node")

        if clicked_node == center_node:
            # Toggle off
            return {
                "center_node": None,
                "distance_info": {},
                "max_distance": max_distance,
            }
        else:
            # Select new center
            distance_info = get_nodes_within_distance(
                graph, clicked_node, max_distance
            )
            return {
                "center_node": clicked_node,
                "distance_info": distance_info,
                "max_distance": max_distance,
            }

    # NEW: Save dragged positions from Cytoscape
    @app.callback(
        Output("cytoscape-graph", "elements", allow_duplicate=True),
        Input("cytoscape-graph", "elements"),
        prevent_initial_call=True,
    )
    def save_cytoscape_positions(elements):
        """
        Persists node positions after dragging in Cytoscape.
        """
        if not elements:
            raise PreventUpdate

        if not isinstance(getattr(current_app, "node_positions", {}), dict):
            current_app.node_positions = {}

        for el in elements:
            if "position" in el and "data" in el:
                node_id = el["data"].get("id")
                x = el["position"]["x"] / 500  # unscale
                y = el["position"]["y"] / 500
                current_app.node_positions[node_id] = (x, y)

        raise PreventUpdate

    # 4) Render the graph and handle exploration interactions (REMOVE drag_mode parameter)
    @app.callback(
        Output("network-graph", "figure"),
        Output("exploration-state", "data"),
        Input("graph-store", "data"),
        Input("layout-dropdown", "value"),
        Input("options-checklist", "value"),
        Input("node-size-slider", "value"),
        Input("distance-slider", "value"),
        Input("reset-btn", "n_clicks"),
        Input("clear-btn", "n_clicks"),
        Input("network-graph", "clickData"),
        Input("drag-mode-store", "data"),
        State("exploration-state", "data"),
        prevent_initial_call=True,
    )
    def update_graph_view(
        graph_data,
        layout_method,
        options,
        node_size_range,
        max_distance,
        reset_clicks,
        clear_clicks,
        click_data,
        drag_mode_data,
        exploration_state,
    ):
        """
        Main graph update handler for Plotly view.
        """
        if not graph_data:
            fig = go.Figure()
            fig.update_layout(
                title="Select a graph to begin exploration",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="white",
                height=800,
            )
            return fig, exploration_state

        graph_path = graph_data.get("path")
        title = graph_data.get("title", "Interactive Network Graph")
        if not graph_path:
            raise PreventUpdate

        graph = load_graph_from_path(graph_path, current_app.graph_cache)
        if graph is None:
            return (
                go.Figure(
                    layout={
                        "title": f"Error loading graph: {os.path.basename(graph_path)}"
                    }
                ),
                exploration_state,
            )

        show_labels = "labels" in (options or [])
        force_refresh = "refresh" in (options or [])

        triggered_id = (
            ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        )

        center_node = exploration_state.get("center_node")
        distance_info = exploration_state.get("distance_info", {})

        if triggered_id == "clear-btn":
            center_node = None
            distance_info = {}
        elif triggered_id == "reset-btn":
            cache_key = _layout_cache_key(graph, layout_method)
            layout_cache = get_app_cache_dict("layout_cache")
            if cache_key in layout_cache:
                del layout_cache[cache_key]
            current_app.distance_cache = {}
            center_node = None
            distance_info = {}
        elif triggered_id == "network-graph" and click_data:
            if click_data.get("points"):
                point = click_data["points"][0]
                if "customdata" in point:
                    clicked_node = point["customdata"]
                    if isinstance(clicked_node, list) and clicked_node:
                        clicked_node = clicked_node[0]
                    if clicked_node in graph.nodes():
                        if clicked_node == center_node:
                            center_node = None
                            distance_info = {}
                        else:
                            center_node = clicked_node
                            distance_info = get_nodes_within_distance(
                                graph, clicked_node, max_distance
                            )
        elif triggered_id == "distance-slider" and center_node:
            distance_info = get_nodes_within_distance(
                graph, center_node, max_distance
            )

        if force_refresh:
            cache_key = _layout_cache_key(graph, layout_method)
            layout_cache = get_app_cache_dict("layout_cache")
            if cache_key in layout_cache:
                del layout_cache[cache_key]

        selected_nodes = [center_node] if center_node else []

        fig = create_interactive_figure(
            graph,
            title,
            layout_method,
            show_labels,
            tuple(node_size_range),
            selected_nodes,
            distance_info if distance_info else None,
        )

        new_exploration_state = {
            "center_node": center_node,
            "distance_info": distance_info,
            "max_distance": max_distance,
        }
        return fig, new_exploration_state

    # Clientside callback for handling node drag
    app.clientside_callback(
        """
        function(selectedData) {
            if (!selectedData || !selectedData.points || selectedData.points.length === 0) {
                return window.dash_clientside.no_update;
            }
            const point = selectedData.points[0];
            if (!point.customdata || point.x === undefined || point.y === undefined) {
                return window.dash_clientside.no_update;
            }
            const nodeId = Array.isArray(point.customdata) ? point.customdata[0] : point.customdata;
            return {node: nodeId, x: point.x, y: point.y};
        }
        """,
        Output("dragged-node-store", "data"),
        Input("network-graph", "selectedData"),
        prevent_initial_call=True,
    )

    # # 5) Stats panel
    # @app.callback(
    #     Output("network-graph", "relayoutData"),
    #     Input("network-graph", "selectedData"),
    #     prevent_initial_call=True,
    # )
    # def store_dragged_positions(selected_data):
    #     if not isinstance(selected_data, dict):
    #         raise PreventUpdate
    #     updated = False
    #     for pt in selected_data.get("points", []):
    #         if _update_node_position_from_drag(pt):
    #             updated = True
    #     # Zwracamy `dash.no_update`, aby nie modyfikować wykresu.
    #     if not updated:
    #         raise PreventUpdate
    #     return dash.no_update

    @app.callback(
        Output("cytoscape-legend", "children"),
        Input("cytoscape-graph", "elements"),
        prevent_initial_call=True,
    )
    def build_cytoscape_legend(_):
        """Tworzy legendę po prawej stronie – używa mapy zapamiętanej w update_cytoscape_graph."""
        legend_map = getattr(current_app, "_cyto_legend_map", {})
        if not legend_map:
            return ""

        # sortujemy po dolnej granicy przedziału (wartość rosnąca)
        sorted_items = sorted(legend_map.items(), key=lambda kv: kv[1])

        legend_children = []
        for color, label in sorted_items:
            legend_children.append(
                html.Div(
                    className="cyto-legend-item",
                    children=[
                        html.Div(
                            style={"backgroundColor": color},
                            className="cyto-legend-swatch",
                        ),
                        html.Span(label),
                    ],
                )
            )
        return html.Div(legend_children, className="cytoscape-legend-container")
