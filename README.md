# Graph Visualizer

An interactive web application for exploring and visualizing large graphs. 
It loads graphs serialized with Python’s pickle (NetworkX graphs), 
provides quick browsing of graph collections, and offers responsive 
visualization with cached computations for better performance.

## Live version

The visualizer is currently available at: https://graph.playground.radlab.dev/

Feel free to browse the information and discover trends!

## Key Features

- Load NetworkX graphs saved as pickle files from organized collections.
- Interactive visualization powered by Dash and Plotly.
- Caching of loaded graphs, computed layouts, node positions, and distances to speed up repeated operations.
- Simple, extensible architecture with clear separation of UI layout and callback logic.

## Project Structure

- `app/` or root application files
    - Application bootstrap and configuration
    - UI layout definition
    - Interactive callbacks (e.g., loading a graph, computing/choosing layouts, updating the figure)
- `data/` (recommended)
    - A base folder that contains subfolders (“collections”) of graphs.
    - Each collection can have nested folders and multiple graph files.
    - Graphs should be NetworkX objects serialized with `pickle`.
    - A placeholder info file may be present to describe expected content.
- `ui/`
    - Components and layout factory for the Dash interface.
- `callbacks/`
    - Callback registrations and handlers for user interactions (e.g., selectors, buttons, sliders).
- Other supporting modules
    - Utilities for scanning collections, locating graph files, loading graphs, and simple in-memory caches.

Note: Folder names may vary slightly depending on your setup, but the conceptual responsibilities remain the same.

## Data Organization

- Choose a base folder (e.g., `data/`) to store your graph collections.
- Inside the base folder, create one subdirectory per collection (e.g., `data/social/`, `data/transport/`).
- Within each collection, place one or more pickle files that contain NetworkX graphs. Nested subfolders are supported.
- The app will scan collections and list available graphs for selection.

## Requirements

- Python 3.10+
- Common Python scientific/visualization stack (notably Dash, Plotly, NetworkX)

If your environment is not yet set up, create and activate a virtual environment 
(virtualenv is recommended) and install the necessary packages.

## Getting Started

1. Create and activate a virtual environment
    - macOS/Linux:
        - `python3 -m venv .venv`
        - `source .venv/bin/activate`
    - Windows (PowerShell):
        - `python -m venv .venv`
        - `.venv\Scripts\Activate.ps1`

2. Install dependencies
    - Ensure packages for web UI and graph handling are available (e.g., Dash, Plotly, NetworkX). If you maintain a requirements file, install with `pip install -r requirements.txt`.

3. Prepare your data
    - Organize your graphs as described in “Data Organization”.
    - Verify your pickle files deserialize to NetworkX graphs.

4. Run the app
    - From the `graph_visualiser` directory: `bash run-gunicorn-app.sh`
    - Open the printed local URL in your browser.

## How It Works (High Level)

- The application configures Plotly and starts a Dash server.
- On startup, the UI is constructed and interactive callbacks are registered.
- When you select a collection/graph, the app:
    - Locates files in the chosen collection (including nested directories).
    - Loads the graph from a pickle file, optionally caching it in memory for re-use.
    - Computes or retrieves cached layouts and metrics for fast rendering.
- Visual updates are handled by callbacks, ensuring a responsive experience even as you switch graphs or parameters.

## Caching Strategy

To improve performance:
- Loaded graphs may be cached in memory (especially small/medium ones).
- Generated layouts and node positions are cached to avoid recomputation.
- Derived metrics (e.g., distances) can also be cached for repeated queries.

You can clear caches by restarting the server process.

## Troubleshooting

- If the app can’t find your graphs:
    - Check that your base data directory exists and contains subdirectories with pickle files.
    - Ensure files actually unpickle into NetworkX graphs.
- If visualization is slow:
    - Consider precomputing and caching layouts for large graphs.
    - Split very large collections or files into smaller subsets for iterative exploration.

## Contributing

- Keep UI elements (layout/components) and interactive logic (callbacks) modular.
- Add utility functions for any repeated data operations.
- Use consistent formatting and linting for maintainability.

## License

See the LICENSE file.