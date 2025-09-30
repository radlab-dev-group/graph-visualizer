import os
import glob
import pickle
from typing import List, Dict, Any, Optional
import networkx as nx

# Simple heuristic to decide whether to cache the loaded graph (in-memory)
MAX_CACHE_NODES = 10_000


def get_graph_collections(base_path: str) -> List[str]:
    """
    Scan the base path and return a sorted list of leaf subdirectories (collections),
    represented as relative POSIX-like paths (e.g., 'specific/1').
    A directory that contains subdirectories is not returned itself (only its leaves).
    """
    if not os.path.isdir(base_path):
        print(
            f"Warning: Base data path '{base_path}'"
            f" does not exist or is not a directory."
        )
        return []

    try:
        all_dirs: set[str] = set()
        parents_with_children: set[str] = set()

        for root, dirs, _ in os.walk(base_path):
            # Skip hidden dirs
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            # Mark the current root as a parent if it has children
            rel_root = os.path.relpath(root, base_path)
            rel_root_posix = rel_root.replace(os.sep, "/")
            if dirs and rel_root_posix not in (".", ""):
                parents_with_children.add(rel_root_posix)

            # Collect immediate subdirectories as candidates
            for d in dirs:
                full_dir = os.path.join(root, d)
                rel = os.path.relpath(full_dir, base_path)
                rel_posix = rel.replace(os.sep, "/")
                if rel_posix and rel_posix != ".":
                    all_dirs.add(rel_posix)

        # Leaf directories are those that are not recorded as parents
        leaves = sorted(d for d in all_dirs if d not in parents_with_children)
        return leaves
    except OSError as e:
        print(f"Error scanning directory '{base_path}': {e}")
        return []


def find_graph_files(collection_path: str, pattern: str) -> List[str]:
    """
    Find graph files under the given collection path
    matching the provided glob pattern.
    """
    if not os.path.exists(collection_path):
        print(f"Warning: Path '{collection_path}' does not exist.")
        return []

    search_pattern = os.path.join(collection_path, "**", pattern)
    files = glob.glob(search_pattern, recursive=True)

    print(
        f"Found {len(files)} files in '{collection_path}' "
        f"matching pattern '{pattern}'."
    )
    return files


def load_graph_from_path(path: str, cache: Dict[str, Any]) -> Optional[nx.Graph]:
    """
    Load a NetworkX graph from a pickle file, using a simple in-memory cache.
    """
    if path in cache:
        print(f"Loading graph from cache: {os.path.basename(path)}")
        cached = cache[path]
        if isinstance(cached, nx.Graph):
            return cached
        else:
            # Fallback if cache entry is unexpected
            print(
                f"Warning: Cache entry for '{path}' is not a NetworkX graph. "
                f"Reloading from disk."
            )

    print(f"Loading graph from file: {os.path.basename(path)}")
    try:
        with open(path, "rb") as f:
            graph = pickle.load(f)

        if not isinstance(graph, nx.Graph):
            print(f"Error: Object loaded from '{path}' is not a NetworkX graph.")
            return None

        # Cache small/medium graphs to speed up repeated loads
        try:
            if len(list(graph.nodes())) < MAX_CACHE_NODES:
                cache[path] = graph
        except Exception:
            # If graph-like but missing nodes(), skip caching
            pass

        print(
            f"Loaded graph from '{path}': {len(list(graph.nodes()))} nodes, "
            f"{len(graph.edges())} edges."
        )
        return graph

    except FileNotFoundError:
        print(f"Error: File not found at '{path}'.")
        return None
    except pickle.UnpicklingError:
        print(
            f"Error: Could not unpickle graph from '{path}'. "
            f"The file may be corrupted."
        )
        return None
    except Exception as e:
        print(f"Unexpected error while loading graph from '{path}': {e}")
        return None


def load_graph_path(
    path: str,
    cache: Optional[Dict[str, Any]] = None,
) -> Optional[nx.Graph]:
    """
    Convenience wrapper around :func:`load_graph_from_path`.

    The original callbacks (e.g. the Cytoscape‑related one) sometimes call
    ``load_graph_path`` instead of ``load_graph_from_path``.  This helper
    mirrors the old signature:

    * ``path`` – absolute or relative path to a pickle file that stores a
      ``networkx.Graph`` (or subclass).

    * ``cache`` – optional explicit cache dictionary.  If omitted the function
      falls back to the Flask‑application‑wide cache stored on
      ``current_app.graph_cache`` (created by ``register_callbacks``).

    The function simply forwards the call to ``load_graph_from_path`` and
    returns the loaded ``nx.Graph`` (or ``None`` on failure).  Keeping the
    wrapper separate makes the intent clear and avoids having to modify the
    existing callback code.
    """
    # If the caller did not provide a cache, use the global Flask cache.
    if cache is None:
        # ``graph_cache`` is guaranteed to exist – it is initialised in
        # ``graph_callbacks.register_callbacks``.
        cache = getattr(current_app, "graph_cache", {})

    return load_graph_from_path(path, cache)
