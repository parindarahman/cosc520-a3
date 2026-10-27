"""Unit tests for graph.py 
"""

import pytest

from graph import Graph


def test_new_graph_has_no_edges():
    """A freshly created graph has its vertices but no edges."""
    g = Graph(3)
    assert g.num_vertices == 3
    assert g.num_edges == 0
    assert g.adj == {1: set(), 2: set(), 3: set()}


def test_add_edge_is_undirected():
    """Adding (u, v) makes u and v neighbours of each other."""
    g = Graph(2)
    g.add_edge(1, 2)
    assert 2 in g.adj[1]
    assert 1 in g.adj[2]
    assert g.num_edges == 1


def test_duplicate_and_reversed_edges_are_ignored():
    """(1, 2), (1, 2) again and (2, 1) all count as one edge."""
    g = Graph(2)
    g.add_edge(1, 2)
    g.add_edge(1, 2)
    g.add_edge(2, 1)
    assert g.num_edges == 1


def test_self_loop_is_ignored():
    """An edge from a vertex to itself is not added."""
    g = Graph(2)
    g.add_edge(1, 1)
    assert g.num_edges == 0
    assert g.adj[1] == set()


def test_from_dimacs_reads_header_and_edges(tmp_path):
    """A small DIMACS file is parsed into the right vertex and edge counts."""
    path = tmp_path / "small.clq"
    path.write_text("c a comment\np edge 3 2\ne 1 2\ne 2 3\n")
    g = Graph.from_dimacs(str(path))
    assert g.num_vertices == 3
    assert g.num_edges == 2
    assert g.adj[2] == {1, 3}


def test_from_dimacs_without_header_raises(tmp_path):
    """A DIMACS file with no 'p' line is rejected."""
    path = tmp_path / "bad.clq"
    path.write_text("e 1 2\n")
    with pytest.raises(ValueError):
        Graph.from_dimacs(str(path))


def test_from_mtx_reads_size_line_and_edges(tmp_path):
    """A small Matrix Market file is parsed, skipping '%' comment lines."""
    path = tmp_path / "small.mtx"
    path.write_text(
        "%%MatrixMarket matrix coordinate pattern symmetric\n"
        "% a comment\n"
        "3 3 2\n"
        "2 1\n"
        "3 2\n"
    )
    g = Graph.from_mtx(str(path))
    assert g.num_vertices == 3
    assert g.num_edges == 2
    assert g.adj[2] == {1, 3}


def test_from_mtx_without_size_line_raises(tmp_path):
    """An .mtx file containing only comments is rejected."""
    path = tmp_path / "bad.mtx"
    path.write_text("% only a comment\n")
    with pytest.raises(ValueError):
        Graph.from_mtx(str(path))


def test_from_file_picks_parser_by_extension(tmp_path):
    """from_file reads .mtx and .clq files with the matching parser."""
    mtx = tmp_path / "g.mtx"
    mtx.write_text("2 2 1\n1 2\n")
    clq = tmp_path / "g.clq"
    clq.write_text("p edge 2 1\ne 1 2\n")
    assert Graph.from_file(str(mtx)).num_edges == 1
    assert Graph.from_file(str(clq)).num_edges == 1