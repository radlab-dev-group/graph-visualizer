from dash import Dash
import plotly.io as pio

from ui.layout import create_layout
from callbacks.graph_callbacks import register_callbacks


def configure_plotly() -> None:
    """
    Configure global Plotly settings.
    """
    pio.templates.default = "plotly_white"


def init_server_caches(cache_server) -> None:
    """
    Attach simple in-memory caches to the Flask server object.
    """
    # Cache for loaded graphs
    cache_server.graph_cache = {}
    # Cache for generated graph layouts
    cache_server.layout_cache = {}
    # Cache for node positions
    cache_server.node_positions = {}
    # Cache for computed distances
    cache_server.distance_cache = {}


def create_app() -> Dash:
    """
    Application factory: creates and configures the Dash app.
    """
    configure_plotly()

    apd_p = Dash(__name__, suppress_callback_exceptions=True)
    apd_p.title = "Graph Visualizer"

    # Expose the underlying Flask server and attach caches
    server_p = apd_p.server
    init_server_caches(server_p)

    # Set the layout before registering callbacks
    apd_p.layout = create_layout()

    # Register callbacks; keep app context if callbacks use Flask context
    with server_p.app_context():
        register_callbacks(apd_p)

    return apd_p


# WSGI (Flask) for production
dash_app = create_app()
server = dash_app.server

# Entrypoint for local development
if __name__ == "__main__":
    dash_app.run(debug=True)
