"""
"""

from graph import Graph
from vertex_cover import basic_vertex_cover, is_vertex_cover


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