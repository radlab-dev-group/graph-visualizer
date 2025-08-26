from dash import dcc, html


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
                            dcc.Dropdown(
                                id="graph-selector",
                                options=[],
                                value=None,
                                placeholder="Select a graph to visualize...",
                                className="dropdown",
                                disabled=True,
                            ),
                            html.Div(
                                id="graph-load-status",
                                className="graph-load-status",
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
    """Control panel with layout, distance, node size, options, and action buttons."""
    return html.Div(
        [
            # Layout controls
            html.Div(
                [
                    html.Label("Graph Layout:", className="layout-label"),
                    dcc.Dropdown(
                        id="layout-dropdown",
                        options=[
                            {"label": "Force-directed", "value": "spring"},
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
            # Node size controls
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
            # Options
            html.Div(
                [
                    dcc.Checklist(
                        id="options-checklist",
                        options=[
                            {"label": " Labels", "value": "labels"},
                            {"label": " Force refresh", "value": "refresh"},
                        ],
                        value=["labels"],
                        className="checklist",
                        inline=True,
                    )
                ],
                className="inline-middle",
            ),
            # Buttons
            html.Div(
                [
                    html.Button(
                        "Load Graph", id="reset-btn", className="btn btn-primary"
                    ),
                    html.Button("Clear", id="clear-btn", className="btn btn-danger"),
                    html.Button(
                        "Export", id="export-btn", className="btn btn-success"
                    ),
                ],
                className="buttons-container",
            ),
        ],
        className="control-panel",
    )


def _graph_with_loading() -> dcc.Loading:
    """
    Graph area with a loading overlay and tuned mode-bar configuration.
    """
    return dcc.Loading(
        id="loading",
        type="default",
        children=[
            dcc.Graph(
                id="network-graph",
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "lasso2d",
                        "autoScale2d",
                        "hoverClosestCartesian",
                        "hoverCompareCartesian",
                        "toggleSpikelines",
                    ],
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "distance_network_graph",
                        "height": 1200,
                        "width": 1600,
                        "scale": 2,
                    },
                    "scrollZoom": True,
                    "doubleClick": "reset+autosize",
                    "showTips": True,
                    "watermark": False,
                    "responsive": True,
                },
                className="graph-style",
            )
        ],
    )


def _aux_stores_and_outputs() -> html.Div:
    """
    Hidden stores and output containers.
    """
    return html.Div(
        [
            # Statistics
            html.Div(id="graph-stats", className="stats-container"),
            # Hidden stores
            dcc.Download(id="download-graph-data"),
            dcc.Store(id="graph-store"),
            dcc.Store(
                id="exploration-state",
                data={"center_node": None, "distance_info": {}, "max_distance": 3},
            ),
            dcc.Store(id="graph-paths-store"),
        ]
    )


def create_layout() -> html.Div:
    """
    Application layout with collection selection, controls, graph canvas, and stores.
    """
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
