"""Unit tests for vertex_cover.py.
"""

import random

from graph import Graph
from vertex_cover import (
    basic_vertex_cover,
    is_vertex_cover,
    randomized_vertex_cover,
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


# ---------- is_vertex_cover ----------

def test_is_vertex_cover_accepts_valid_cover():
    """On the path 1-2-3, vertex 2 alone covers both edges."""
    g = make_graph(3, [(1, 2), (2, 3)])
    assert is_vertex_cover(g, {2})


def test_is_vertex_cover_rejects_invalid_cover():
    """On the path 1-2-3, vertex 1 alone leaves edge (2, 3) uncovered."""
    g = make_graph(3, [(1, 2), (2, 3)])
    assert not is_vertex_cover(g, {1})


def test_empty_set_covers_graph_with_no_edges():
    """A graph with no edges is covered by the empty set."""
    g = make_graph(3, [])
    assert is_vertex_cover(g, set())


# ---------- basic_vertex_cover ----------

def test_basic_no_edges_gives_empty_cover():
    """With nothing to cover, the cover is empty."""
    g = make_graph(4, [])
    assert basic_vertex_cover(g) == set()


def test_basic_single_edge_takes_both_endpoints():
    """One edge: both endpoints are added (optimal would be 1)."""
    g = make_graph(2, [(1, 2)])
    assert basic_vertex_cover(g) == {1, 2}


def test_basic_triangle():
    """Triangle: the first edge adds 2 vertices, which covers all 3 edges."""
    g = make_graph(3, [(1, 2), (2, 3), (1, 3)])
    cover = basic_vertex_cover(g)
    assert cover == {1, 2}
    assert is_vertex_cover(g, cover)


def test_basic_path_hits_the_factor_two_bound():
    """Path 1-2-3-4: picks (1,2) then (3,4), so 4 vertices vs. optimal 2."""
    g = make_graph(4, [(1, 2), (2, 3), (3, 4)])
    cover = basic_vertex_cover(g)
    assert cover == {1, 2, 3, 4}
    assert len(cover) <= 2 * 2  # optimal cover {2, 3} has size 2


def test_basic_star_covers_with_centre_and_one_leaf():
    """Star with centre 1: the first edge covers every other edge."""
    g = make_graph(6, [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)])
    cover = basic_vertex_cover(g)
    assert cover == {1, 2}
    assert is_vertex_cover(g, cover)


def test_basic_cover_size_is_always_even():
    """Vertices are added in pairs, so the cover size is even."""
    g = make_graph(6, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (1, 4)])
    assert len(basic_vertex_cover(g)) % 2 == 0


def test_basic_is_valid_on_complete_bipartite_graph():
    """K(3,3): a valid cover within 2x optimal (optimal is one side, size 3)."""
    left, right = [1, 2, 3], [4, 5, 6]
    g = make_graph(6, [(u, v) for u in left for v in right])
    cover = basic_vertex_cover(g)
    assert is_vertex_cover(g, cover)
    assert len(cover) <= 2 * 3


def test_basic_is_valid_on_disconnected_graph():
    """Two separate edges plus an isolated vertex are all handled."""
    g = make_graph(5, [(1, 2), (3, 4)])
    cover = basic_vertex_cover(g)
    assert cover == {1, 2, 3, 4}
    assert 5 not in cover


# ---------- randomized_vertex_cover ----------

SEEDS = range(20)  # many seeds, so every random branch gets exercised


def test_randomized_no_edges_gives_empty_cover():
    """With nothing to cover, the cover is empty."""
    g = make_graph(4, [])
    assert randomized_vertex_cover(g, random.Random(0)) == set()


def test_randomized_single_edge_takes_exactly_one_endpoint():
    """One edge: exactly one endpoint is added, never both (Pitt step 3)."""
    g = make_graph(2, [(1, 2)])
    for seed in SEEDS:
        cover = randomized_vertex_cover(g, random.Random(seed))
        assert len(cover) == 1
        assert cover <= {1, 2}


def test_randomized_single_edge_can_pick_either_endpoint():
    """Across seeds, both u and v get chosen, so the coin really is used."""
    g = make_graph(2, [(1, 2)])
    picks = {frozenset(randomized_vertex_cover(g, random.Random(s)))
             for s in SEEDS}
    assert picks == {frozenset({1}), frozenset({2})}


def test_randomized_always_returns_valid_cover():
    """Every run, on several shapes of graph, covers all edges."""
    graphs = [
        make_graph(3, [(1, 2), (2, 3), (1, 3)]),                  # triangle
        make_graph(4, [(1, 2), (2, 3), (3, 4)]),                  # path
        make_graph(6, [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]),  # star
        make_graph(6, [(u, v) for u in (1, 2, 3) for v in (4, 5, 6)]),  # K(3,3)
        make_graph(5, [(1, 2), (3, 4)]),                          # disconnected
    ]
    for g in graphs:
        for seed in SEEDS:
            assert is_vertex_cover(g, randomized_vertex_cover(g, random.Random(seed)))


def test_randomized_same_seed_gives_same_cover():
    """A fixed seed makes the result reproducible."""
    g = make_graph(6, [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (1, 4)])
    first = randomized_vertex_cover(g, random.Random(7))
    second = randomized_vertex_cover(g, random.Random(7))
    assert first == second


def test_randomized_adds_one_vertex_per_chosen_edge():
    """Each uncovered edge adds one vertex, so on disjoint edges size = #edges."""
    g = make_graph(6, [(1, 2), (3, 4), (5, 6)])
    for seed in SEEDS:
        assert len(randomized_vertex_cover(g, random.Random(seed))) == 3


def test_randomized_only_uses_vertices_of_the_graph():
    """The cover never contains isolated vertices or labels outside the graph."""
    g = make_graph(5, [(1, 2), (2, 3)])  # vertices 4 and 5 are isolated
    for seed in SEEDS:
        assert randomized_vertex_cover(g, random.Random(seed)) <= {1, 2, 3}


def test_randomized_star_can_find_optimal_cover():
    """Star: picking the centre covers everything, so some seed finds the
    optimal cover {1}; no run adds more vertices than there are edges."""
    g = make_graph(6, [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)])
    sizes = [len(randomized_vertex_cover(g, random.Random(s))) for s in SEEDS]
    assert min(sizes) == 1      # some run finds the optimal cover {1}
    assert max(sizes) <= 5      # never more vertices than the graph has edges