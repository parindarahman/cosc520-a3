"""
"""

import sys

from graph import Graph


def basic_vertex_cover(graph: Graph) -> set:
    """
    Deterministic 2-approximation (Algorithm VC2, Salavatipour 2013, sec. 1.4).

    Scans the edges in a fixed order. For each edge with neither endpoint in
    the cover yet, adds BOTH endpoints. Skipping already-covered edges is
    equivalent to deleting every edge incident to the chosen vertices.
    The chosen edges form a maximal matching M, so |cover| = 2|M| <= 2 * OPT.

    Input: an undirected Graph.
    Output: a set of vertices that covers every edge.
    Time: O(V + E). Extra space: O(V) for the cover.
    """
    cover = set()
    for u, v in graph.edges:
        if u in cover or v in cover:
            continue  # edge already covered ("deleted")
        cover.add(u)
        cover.add(v)
    return cover


def is_vertex_cover(graph: Graph, cover: set) -> bool:
    """
    Check whether a set of vertices is a valid vertex cover.

    Input: a Graph and a candidate set of vertices.
    Output: True if every edge has at least one endpoint in cover.
    Time: O(E).
    """
    return all(u in cover or v in cover for u, v in graph.edges)


if __name__ == "__main__":
    # Quick demo: python vertex_cover.py data/frb30-15-1.mtx
    if len(sys.argv) != 2:
        print("Usage: python vertex_cover.py <path-to-graph-file>")
        sys.exit(1)

    g = Graph.from_file(sys.argv[1])
    result = basic_vertex_cover(g)
    print(f"Vertices: {g.num_vertices}, edges: {g.num_edges}")
    print(f"Basic cover size: {len(result)}")
    print(f"Valid cover: {is_vertex_cover(g, result)}")