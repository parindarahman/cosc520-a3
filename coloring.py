"""Greedy graph coloring with different vertex orderings.
"""

import random
import sys

from graph import Graph


def greedy_coloring(graph: Graph, order: list) -> dict:
    """
    Greedy coloring (GreedyBasicGraphColoring from the assignment).

    Visits the vertices in the given order and gives each one the lowest
    color number (0, 1, 2, ...) not used by any of its already-colored
    neighbours. A vertex with d neighbours can block at most d colors, so
    the search for a free color takes at most d + 1 steps.

    Input: an undirected Graph, and a list containing every vertex exactly
           once, in the order to color them.
    Output: a dict mapping each vertex to its color (an int >= 0).
    Time: O(V + E). Space: O(V) for the colors, plus O(max degree)
          for the set of neighbour colors.
    """
    colors = {}
    for v in order:
        # Colors already taken by v's colored neighbours (at most deg(v) of them).
        used = {colors[w] for w in graph.adj[v] if w in colors}
        # Lowest legal color: count up from 0 until a free one is found.
        color = 0
        while color in used:
            color += 1
        colors[v] = color
    return colors


def basic_greedy_coloring(graph: Graph) -> dict:
    """
    Basic greedy coloring: vertices in their natural order 1, 2, ..., n.

    Input: an undirected Graph.
    Output: a dict mapping each vertex to its color.
    Time: O(V + E).
    """
    natural_order = list(range(1, graph.num_vertices + 1))
    return greedy_coloring(graph, natural_order)


def random_order_coloring(graph: Graph, rng: random.Random) -> dict:
    """
    Greedy coloring with the vertices in a uniformly random order.

    A true "arbitrary order" baseline: the natural order 1..n can follow
    structure in how a file numbers its vertices (FRB files list vertices
    group by group), which a random order does not.

    Input: an undirected Graph and a seeded random number generator.
    Output: a dict mapping each vertex to its color.
    Time: O(V + E) (shuffle O(V) + greedy O(V + E)).
    """
    order = list(range(1, graph.num_vertices + 1))
    rng.shuffle(order)
    return greedy_coloring(graph, order)


def smallest_last_order(graph: Graph) -> list:
    """
    Smallest Last vertex ordering (Matula & Beck, 1983).

    Repeatedly removes a vertex of minimum degree in the remaining graph
    (updating its neighbours' degrees), then returns the vertices in REVERSE
    removal order. A vertex removed early had few remaining neighbours, so
    coloring it last is safe: it always finds a free color. Greedy coloring
    in this order uses at most d + 1 colors, where d (the degeneracy) is the
    largest minimum degree seen during the removals.

    Degree buckets make each step O(1) amortised: bucket[k] holds the
    remaining vertices of current degree k, and the pointer to the smallest
    non-empty bucket only drops by at most 1 per removal.

    Input: an undirected Graph.
    Output: a list of all vertices, in the order to color them.
    Time: O(V + E). Space: O(V) for degrees, buckets and the order.
    """
    degree = {v: len(graph.adj[v]) for v in graph.adj}
    max_degree = max(degree.values(), default=0)
    buckets = [set() for _ in range(max_degree + 1)]
    for v, d in degree.items():
        buckets[d].add(v)

    removed = set()
    removal_order = []
    min_degree = 0
    for _ in range(graph.num_vertices):
        while not buckets[min_degree]:  # find the smallest non-empty bucket
            min_degree += 1
        v = buckets[min_degree].pop()
        removed.add(v)
        removal_order.append(v)
        for w in graph.adj[v]:  # each edge is handled once, from its first-removed end
            if w not in removed:
                buckets[degree[w]].remove(w)
                degree[w] -= 1
                buckets[degree[w]].add(w)
        # A neighbour's degree fell by 1, so the minimum can drop by at most 1.
        min_degree = max(min_degree - 1, 0)

    removal_order.reverse()
    return removal_order


def smallest_last_coloring(graph: Graph) -> dict:
    """
    Greedy coloring using the Smallest Last vertex ordering.

    Input: an undirected Graph.
    Output: a dict mapping each vertex to its color.
    Time: O(V + E) (ordering O(V + E) + greedy O(V + E)).
    """
    return greedy_coloring(graph, smallest_last_order(graph))


def is_valid_coloring(graph: Graph, colors: dict) -> bool:
    """
    Check that every vertex is colored and no edge joins two equal colors.

    Input: a Graph and a dict mapping vertices to colors.
    Output: True if the coloring is complete and proper.
    Time: O(V + E).
    """
    if len(colors) != graph.num_vertices:
        return False
    return all(colors[u] != colors[v] for u, v in graph.edges)


def count_colors(colors: dict) -> int:
    """
    Input: a dict mapping vertices to colors.
    Output: the number of distinct colors used.
    """
    return len(set(colors.values()))


if __name__ == "__main__":
    # Quick demo: python coloring.py data/frb30-15-1.mtx
    if len(sys.argv) != 2:
        print("Usage: python coloring.py <path-to-graph-file>")
        sys.exit(1)

    g = Graph.from_file(sys.argv[1])
    print(f"Vertices: {g.num_vertices}, edges: {g.num_edges}")

    basic = basic_greedy_coloring(g)
    print(f"Basic greedy colors: {count_colors(basic)} "
          f"(valid: {is_valid_coloring(g, basic)})")

    random_order = random_order_coloring(g, random.Random(0))
    print(f"Random-order colors: {count_colors(random_order)} "
          f"(valid: {is_valid_coloring(g, random_order)})")

    smallest_last = smallest_last_coloring(g)
    print(f"Smallest Last colors: {count_colors(smallest_last)} "
          f"(valid: {is_valid_coloring(g, smallest_last)})")