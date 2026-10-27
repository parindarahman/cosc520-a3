"""Approximation algorithms for the (unweighted) vertex cover problem.
"""

import random
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


def randomized_vertex_cover(graph: Graph, rng: random.Random) -> set:
    """
    Randomized 2-approximation in expectation (Algorithm U, Pitt 1985,
    Yale TR-404, sec. 3).

    Scans the edges in a fixed order. For each edge with neither endpoint in
    the cover yet, flips a fair coin and adds ONE endpoint (u or v).
    Each flip picks a vertex of an optimal cover C with probability >= 1/2,
    so the expected cover size is at most 2 * |C|.

    Input: an undirected Graph and a seeded random number generator
           (e.g. random.Random(42)), so runs are reproducible.
    Output: a set of vertices that covers every edge.
    Time: O(V + E). Extra space: O(V) for the cover.
    """
    cover = set()
    for u, v in graph.edges:  # step 1: edges in a fixed, arbitrary order
        if u in cover or v in cover:
            continue  # step 2: skip edges that are already covered
        cover.add(rng.choice((u, v)))  # step 3: fair coin picks u or v
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
    print(f"Vertices: {g.num_vertices}, edges: {g.num_edges}")

    basic = basic_vertex_cover(g)
    print(f"Basic cover size: {len(basic)} (valid: {is_vertex_cover(g, basic)})")

    randomized = randomized_vertex_cover(g, random.Random(42))
    print(f"Randomized cover size: {len(randomized)} "
          f"(valid: {is_vertex_cover(g, randomized)})")