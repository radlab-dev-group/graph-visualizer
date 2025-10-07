# ui/layout.py
from dash import dcc, html
import dash_cytoscape as cyto


def _top_bar() -> html.Div:
    """Top bar with collection and graph selectors."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Select Collection:", className="label-bold"),
                            dcc.Dropdown(
                                id="collection-selector",
                                options=[],
                                placeholder="Select a collection (subfolder in data/)...",
                                className="dropdown",
                                clearable=True,
                            ),
                        ],
                    ),
                    html.Div(
                        [
                            html.Label("Select Graph:", className="label-bold"),
                            # Spinner while loading the list of graphs (options/value/disabled)
                            dcc.Loading(
                                id="graph-selector-loading",
                                type="dot",
                                children=[
                                    dcc.Dropdown(
                                        id="graph-selector",
                                        options=[],
                                        value=None,
                                        placeholder="Select a graph to visualize...",
                                        className="dropdown",
                                        disabled=True,
                                    )
                                ],
                            ),
                            # Spinner while loading the selected graph (status + cache)
                            dcc.Loading(
                                id="graph-load-status-loading",
                                type="circle",
                                children=[
                                    html.Div(
                                        id="graph-load-status",
                                        className="graph-load-status",
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
                className="top-bar-grid",
            )
        ],
        className="top-bar-inner",
    )


def _controls_panel() -> html.Div:
    """Control panel with layout, distance, node size, options and action buttons."""
    return html.Div(
        [
            # Layout controls
            html.Div(
                [
                    html.Label("Graph Layout:", className="layout-label"),
                    dcc.Dropdown(
                        id="layout-dropdown",
                        options=[
                            {"label": "Force‑directed", "value": "spring"},
                            {"label": "Circular", "value": "circular"},
                            {"label": "Random", "value": "random"},
                            {"label": "Shell", "value": "shell"},
                        ],
                        value="spring",
                        className="layout-dropdown",
                        clearable=False,
                    ),
                ],
                className="inline-top",
            ),
            # Distance controls
            html.Div(
                [
                    html.Label("Max Distance To Show:", className="distance-label"),
                    dcc.Slider(
                        id="distance-slider",
                        min=1,
                        max=10,
                        step=1,
                        marks={i: str(i) for i in range(1, 11)},
                        value=3,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                className="distance-container",
            ),
            # Node‑size controls
            html.Div(
                [
                    html.Label("Display Node Size:", className="node-size-label"),
                    dcc.RangeSlider(
                        id="node-size-slider",
                        min=1,
                        max=75,
                        step=5,
                        marks={1: "1", 25: "25", 50: "50", 75: "75"},
                        value=[50, 75],
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
                className="node-size-container",
            ),
            # Options checklist
            html.Div(
                [
                    dcc.Checklist(
                        id="options-checklist",
                        options=[
                            {"label": " Labels", "value": "labels"},
                            {"label": " Force refresh", "value": "refresh"},
                        ],
                        value=["labels", "refresh"],
                        className="checklist",
                        inline=True,
                    )
                ],
                className="inline-middle",
            ),
            # Action buttons
            html.Div(
                [
                    html.Button(
                        "Load Graph", id="reset-btn", className="btn btn-primary"
                    ),
                    html.Button("Clear", id="clear-btn", className="btn btn-danger"),
                    html.Button(
                        "🔀 Toggle Drag Mode",
                        id="toggle-drag-btn",
                        className="btn btn-success",
                    ),
                ],
                className="buttons-container",
            ),
        ],
        className="control-panel",
    )


def _graph_with_loading() -> dcc.Loading:
    """
    Returns the loading‑wrapper that contains:
      • the Plotly figure (id="network‑graph")
      • the Cytoscape component (id="cytoscape‑graph")
      • the legend (id="cytoscape‑legend") placed on the **right‑hand side**
    The layout is a simple flex‑container so the legend stays next to the
    graph exactly as it does in the native Plotly view.
    """
    # ----------------------------------------------------------------------
    #  Cytoscape stylesheet – exactly the same as the one you used before.
    #  It is kept here so the function is self‑contained.
    # ----------------------------------------------------------------------
    cytoscape_stylesheet = [
        # ------------------- nodes -------------------
        {
            "selector": "node",
            "style": {
                "content": "data(label)",
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "12px",
                "color": "#000000",
                # colour is taken from the element’s data (set in the callback)
                "background-color": "data(color)",
                "border-color": "data(color)",
                "border-width": 1,
                "opacity": 0.9,
                "width": "data(size)",
                "height": "data(size)",
            },
        },
        # ------------------- grabbed node -------------------
        {
            "selector": "node:grabbed",
            "style": {
                "background-color": "#FF4136",
                "border-color": "#FF4136",
                "width": "data(size_selected)",
                "height": "data(size_selected)",
            },
        },
        # ------------------- selected node -------------------
        {
            "selector": ".selected",
            "style": {
                "background-color": "#FF4136",
                "border-color": "darkred",
                "border-width": 3,
            },
        },
        # ------------------- distance‑highlighted nodes -------------------
        {
            "selector": ".distance",
            "style": {"background-color": "#FF851B"},
        },
        # ------------------- edges -------------------
        {
            "selector": "edge",
            "style": {
                "width": 1,
                "line-color": "#CCCCCC",
                "opacity": 0.7,
            },
        },
    ]

    # ----------------------------------------------------------------------
    #  Flex container that holds the graph (left) and the legend (right)
    # ----------------------------------------------------------------------
    return dcc.Loading(
        id="loading",
        type="default",
        children=[
            html.Div(
                # Flex‑box – legend will sit on the right side of the graph
                style={"display": "flex", "alignItems": "flex-start"},
                children=[
                    # ------------------------------------------------------------------
                    #  LEFT PANEL – Plotly + Cytoscape (they occupy the same space,
                    #  only one of them is visible at a time – the toggle‑drag
                    #  callback switches the ``display`` style).
                    # ------------------------------------------------------------------
                    html.Div(
                        id="graph-wrapper",
                        # “flex:1” makes this column take all remaining width
                        style={"flex": "1", "minWidth": "0"},
                        children=[
                            # Plotly figure (shown when we are in Plotly mode)
                            dcc.Graph(
                                id="network-graph",
                                style={"width": "100%", "height": "800px"},
                                config={"displayModeBar": False, "scrollZoom": True},
                            ),
                            # Cytoscape – initially hidden, will be shown when the user
                            # clicks the “drag / explore” button.
                            cyto.Cytoscape(
                                id="cytoscape-graph",
                                layout={"name": "preset"},
                                style={
                                    "width": "100%",
                                    "height": "800px",
                                    "display": "none",  # hidden until drag‑mode is on
                                },
                                stylesheet=cytoscape_stylesheet,
                            ),
                        ],
                    ),
                    # ------------------------------------------------------------------
                    #  RIGHT PANEL – legend for Cytoscape mode
                    # ------------------------------------------------------------------
                    html.Div(
                        id="cytoscape-legend",
                        className="cytoscape-legend",
                        # a little margin makes the legend look separated from the graph
                        style={
                            "marginLeft": "20px",
                            "minWidth": "150px",
                            "maxWidth": "250px",
                        },
                    ),
                ],
            )
        ],
    )


def _aux_stores_and_outputs() -> html.Div:
    """Hidden dcc.Store elements and extra output containers."""
    return html.Div(
        [
            html.Div(id="graph-stats", className="stats-container"),
            dcc.Download(id="download-graph-data"),
            dcc.Store(id="graph-store"),
            dcc.Store(
                id="exploration-state",
                data={"center_node": None, "distance_info": {}, "max_distance": 3},
            ),
            dcc.Store(id="graph-paths-store"),
            dcc.Store(id="drag-mode-store", data={"mode": "plotly"}),
        ]
    )


def create_layout() -> html.Div:
    """Kompletny layout aplikacji."""
    return html.Div(
        [
            html.H1("🔗 Network Explorer", className="title"),
            html.Div([_top_bar()], className="top-bar"),
            _controls_panel(),
            _graph_with_loading(),
            _aux_stores_and_outputs(),
        ],
        className="root-container",
    )
