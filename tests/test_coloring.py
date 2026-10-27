"""Unit tests for coloring.py.
"""

import random

from graph import Graph
from coloring import (
    basic_greedy_coloring,
    count_colors,
    greedy_coloring,
    is_valid_coloring,
    random_order_coloring,
    smallest_last_coloring,
    smallest_last_order,
)


def make_graph(num_vertices, edges):
    """
    Build a small test graph.

    Input: number of vertices and a list of (u, v) edges.
    Output: the corresponding Graph.
    """
    g = Graph(num_vertices)
    for u, v in edges:
        g.add_edge(u, v)
    return g


def random_graph(num_vertices, num_edges, seed):
    """
    Build a reproducible random graph for property-style tests.

    Input: vertex count, number of edge attempts, and a seed.
    Output: a Graph (duplicates and self-loops are dropped by add_edge).
    """
    rng = random.Random(seed)
    g = Graph(num_vertices)
    for _ in range(num_edges):
        g.add_edge(rng.randint(1, num_vertices), rng.randint(1, num_vertices))
    return g


def complete_graph(n):
    """Input: n. Output: the complete graph K_n (every pair joined)."""
    return make_graph(n, [(u, v) for u in range(1, n + 1) for v in range(u + 1, n + 1)])


# ---------- is_valid_coloring / count_colors ----------

def test_is_valid_coloring_accepts_proper_coloring():
    """Alternating colors on a path are valid."""
    g = make_graph(3, [(1, 2), (2, 3)])
    assert is_valid_coloring(g, {1: 0, 2: 1, 3: 0})


def test_is_valid_coloring_rejects_clash():
    """Two neighbours with the same color are not valid."""
    g = make_graph(2, [(1, 2)])
    assert not is_valid_coloring(g, {1: 0, 2: 0})


def test_is_valid_coloring_rejects_missing_vertex():
    """Every vertex must be colored."""
    g = make_graph(3, [(1, 2)])
    assert not is_valid_coloring(g, {1: 0, 2: 1})


def test_count_colors():
    """Distinct color values are counted once each."""
    assert count_colors({1: 0, 2: 1, 3: 0, 4: 2}) == 3


# ---------- greedy_coloring ----------

def test_greedy_graph_with_no_edges_uses_one_color():
    """Isolated vertices can all share color 0."""
    g = make_graph(4, [])
    colors = greedy_coloring(g, [1, 2, 3, 4])
    assert set(colors.values()) == {0}


def test_greedy_gives_lowest_free_color():
    """On the path 1-2-3-4 in natural order, colors alternate 0, 1, 0, 1."""
    g = make_graph(4, [(1, 2), (2, 3), (3, 4)])
    assert greedy_coloring(g, [1, 2, 3, 4]) == {1: 0, 2: 1, 3: 0, 4: 1}


def test_greedy_result_depends_on_order():
    """Same path, order 1, 4, 2, 3: a bad order costs an extra color."""
    g = make_graph(4, [(1, 2), (2, 3), (3, 4)])
    assert count_colors(greedy_coloring(g, [1, 2, 3, 4])) == 2
    assert count_colors(greedy_coloring(g, [1, 4, 2, 3])) == 3


def test_greedy_complete_graph_needs_n_colors():
    """In K_5 every vertex touches every other, so 5 colors are needed."""
    g = complete_graph(5)
    colors = greedy_coloring(g, [1, 2, 3, 4, 5])
    assert is_valid_coloring(g, colors)
    assert count_colors(colors) == 5


def test_greedy_never_exceeds_max_degree_plus_one():
    """Greedy uses at most (max degree + 1) colors, for any order."""
    for seed in range(10):
        g = random_graph(40, 150, seed)
        max_degree = max(len(n) for n in g.adj.values())
        order = list(g.adj)
        random.Random(seed).shuffle(order)
        colors = greedy_coloring(g, order)
        assert is_valid_coloring(g, colors)
        assert count_colors(colors) <= max_degree + 1


# ---------- basic_greedy_coloring ----------

def test_basic_is_valid_on_various_graphs():
    """Natural-order greedy gives a proper coloring on several shapes."""
    graphs = [
        make_graph(3, [(1, 2), (2, 3), (1, 3)]),                    # triangle
        make_graph(6, [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]),    # star
        make_graph(6, [(u, v) for u in (1, 2, 3) for v in (4, 5, 6)]),  # K(3,3)
        make_graph(5, [(1, 2), (3, 4)]),                            # disconnected
    ]
    for g in graphs:
        assert is_valid_coloring(g, basic_greedy_coloring(g))


def test_basic_bipartite_natural_order_uses_two_colors():
    """K(3,3) with one side numbered first is 2-colored."""
    g = make_graph(6, [(u, v) for u in (1, 2, 3) for v in (4, 5, 6)])
    assert count_colors(basic_greedy_coloring(g)) == 2


# ---------- random_order_coloring ----------

def test_random_order_is_valid_on_random_graphs():
    """Any random order still gives a proper coloring."""
    for seed in range(10):
        g = random_graph(40, 150, seed)
        assert is_valid_coloring(g, random_order_coloring(g, random.Random(seed)))


def test_random_order_same_seed_gives_same_coloring():
    """A fixed seed makes the result reproducible."""
    g = random_graph(40, 150, 3)
    assert (random_order_coloring(g, random.Random(5))
            == random_order_coloring(g, random.Random(5)))


def test_random_order_varies_with_seed():
    """Different seeds give different orders, so colorings differ somewhere."""
    g = random_graph(40, 150, 3)
    colorings = {tuple(sorted(random_order_coloring(g, random.Random(s)).items()))
                 for s in range(10)}
    assert len(colorings) > 1


# ---------- smallest_last_order ----------

def test_smallest_last_order_contains_every_vertex_once():
    """The ordering is a permutation of the vertices."""
    for seed in range(5):
        g = random_graph(30, 80, seed)
        order = smallest_last_order(g)
        assert sorted(order) == list(range(1, 31))


def test_smallest_last_order_empty_and_edgeless_graphs():
    """Graphs with no vertices or no edges are handled."""
    assert smallest_last_order(Graph(0)) == []
    assert sorted(smallest_last_order(make_graph(3, []))) == [1, 2, 3]


def test_smallest_last_removes_a_minimum_degree_vertex_each_step():
    """Replaying the removal (reverse of the order), each removed vertex has
    the minimum degree among the vertices still remaining."""
    for seed in range(10):
        g = random_graph(30, 90, seed)
        remaining = set(g.adj)
        for v in reversed(smallest_last_order(g)):
            def remaining_degree(x):
                return len(g.adj[x] & remaining)
            assert remaining_degree(v) == min(remaining_degree(x) for x in remaining)
            remaining.remove(v)


def test_smallest_last_star_puts_centre_first_to_color():
    """Star: leaves (degree 1) are removed first, so the centre is colored early."""
    g = make_graph(6, [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)])
    order = smallest_last_order(g)
    assert order.index(1) <= 1  # centre is among the first two colored


# ---------- smallest_last_coloring ----------

def test_smallest_last_coloring_is_valid_on_random_graphs():
    """Smallest Last always gives a proper coloring."""
    for seed in range(10):
        g = random_graph(50, 200, seed)
        assert is_valid_coloring(g, smallest_last_coloring(g))


def test_smallest_last_fixes_the_bad_path_order():
    """On a path, Smallest Last finds the optimal 2 colors."""
    g = make_graph(6, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
    assert count_colors(smallest_last_coloring(g)) == 2


def test_smallest_last_trees_use_at_most_two_colors():
    """Trees are 1-degenerate, so Smallest Last colors them with <= 2 colors."""
    tree = make_graph(8, [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7), (7, 8)])
    assert count_colors(smallest_last_coloring(tree)) <= 2


def test_smallest_last_complete_graph_needs_n_colors():
    """No ordering can beat K_n's n colors."""
    g = complete_graph(6)
    assert count_colors(smallest_last_coloring(g)) == 6