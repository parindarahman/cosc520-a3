"""Undirected graph representation and graph file parsers (DIMACS, .mtx).
"""


class Graph:
    """Undirected simple graph stored as an adjacency list and an edge list."""

    def __init__(self, num_vertices: int):
        """
        Create an empty graph.

        Input: num_vertices - vertices are labelled 1..num_vertices.
        Output: an empty Graph with no edges.
        """
        self.num_vertices = num_vertices
        self.adj = {v: set() for v in range(1, num_vertices + 1)}
        self.edges = []

    @property
    def num_edges(self) -> int:
        """
        Input: none.
        Output: the number of edges in the graph.
        """
        return len(self.edges)

    def add_edge(self, u: int, v: int) -> None:
        """
        Add the undirected edge (u, v), ignoring self-loops and duplicates.

        Input: endpoints u and v (both in 1..num_vertices).
        Output: none; the graph is updated in place.
        """
        if u == v or v in self.adj[u]:
            return
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.edges.append((u, v))

    @classmethod
    def from_dimacs(cls, path: str) -> "Graph":
        """
        Build a graph from a DIMACS-format file (.clq, .col, .mis).

        Input: path to the file. Expected lines are
               'c ...' (comment), 'p edge <n> <m>' (header), 'e <u> <v>' (edge).
        Output: the Graph described by the file.
        Raises: ValueError if the file has no 'p' header line before its edges.
        """
        graph = None
        with open(path) as file:
            for line in file:
                parts = line.split()
                if not parts or parts[0] == "c":
                    continue
                if parts[0] == "p":
                    graph = cls(int(parts[2]))
                elif parts[0] == "e":
                    if graph is None:
                        raise ValueError(f"{path}: edge found before 'p' header")
                    graph.add_edge(int(parts[1]), int(parts[2]))
        if graph is None:
            raise ValueError(f"{path}: no 'p' header line found")
        return graph

    @classmethod
    def from_mtx(cls, path: str) -> "Graph":
        """
        Build a graph from a Matrix Market file (.mtx), the format used by
        Network Repository downloads.

        Input: path to the file. Lines starting with '%' are comments; the
               first other line is '<rows> <cols> <entries>'; each following
               line is '<u> <v>' (an optional third weight column is ignored).
        Output: the Graph described by the file.
        Raises: ValueError if the file has no size line.
        """
        graph = None
        with open(path) as file:
            for line in file:
                parts = line.split()
                if not parts or parts[0].startswith("%"):
                    continue
                if graph is None:
                    graph = cls(int(parts[0]))
                else:
                    graph.add_edge(int(parts[0]), int(parts[1]))
        if graph is None:
            raise ValueError(f"{path}: no size line found")
        return graph

    @classmethod
    def from_file(cls, path: str) -> "Graph":
        """
        Load a graph, choosing the parser from the file extension.

        Input: path to a .mtx file or a DIMACS file (.clq, .col, .mis).
        Output: the Graph described by the file.
        """
        if path.lower().endswith(".mtx"):
            return cls.from_mtx(path)
        return cls.from_dimacs(path)