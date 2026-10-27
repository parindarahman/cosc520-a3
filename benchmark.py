"""Benchmark the vertex cover algorithms on every graph in a data folder.

For each graph file it runs every algorithm several times, checks that each
result is a valid cover, and records running time and cover size. Results are
written to a CSV that plot_results.py turns into plots and a table.

Usage:
    python benchmark.py                      # data/ -> results/, 30 runs
    python benchmark.py --runs 30
    python benchmark.py --data data --out results
"""

import argparse
import csv
import os
import random
import re
import statistics
import time

from graph import Graph
from vertex_cover import basic_vertex_cover, is_vertex_cover, randomized_vertex_cover

# Each algorithm takes (graph, seed). The basic version ignores the seed.
VERTEX_COVER_ALGORITHMS = {
    "basic": lambda graph, seed: basic_vertex_cover(graph),
    "randomized": lambda graph, seed: randomized_vertex_cover(graph, random.Random(seed)),
}

GRAPH_EXTENSIONS = (".mtx", ".clq", ".col", ".mis")

CSV_COLUMNS = [
    "graph", "vertices", "edges", "optimal", "algorithm", "runs",
    "time_median_ms", "time_q1_ms", "time_q3_ms",
    "size_mean", "size_min", "size_max", "ratio_mean",
]


def optimal_cover_size(file_name: str):
    """
    Known optimum for a BHOSLIB/FRB instance, read from its name.

    frbN-K has N groups of K vertices (|V| = N*K) and a maximum independent
    set of size N, so the minimum vertex cover has N*K - N vertices.

    Input: a file name such as 'frb30-15-1.mtx'.
    Output: the optimal cover size, or None if the name is not an FRB name.
    """
    match = re.match(r"frb(\d+)-(\d+)", file_name)
    if match is None:
        return None
    groups, group_size = int(match.group(1)), int(match.group(2))
    return groups * group_size - groups


def time_algorithm(algorithm, graph: Graph, runs: int):
    """
    Run one algorithm several times and measure it.

    Input: an algorithm taking (graph, seed), a Graph, and the number of runs.
           Run i uses seed i, so results are reproducible.
    Output: (list of run times in milliseconds, list of cover sizes).
    Raises: AssertionError if any run returns an invalid cover.
    """
    times_ms, sizes = [], []
    algorithm(graph, 0)  # warm-up run, not timed, so caching doesn't skew run 1
    for seed in range(runs):
        start = time.perf_counter()
        cover = algorithm(graph, seed)
        times_ms.append((time.perf_counter() - start) * 1000)
        assert is_vertex_cover(graph, cover), "algorithm returned an invalid cover"
        sizes.append(len(cover))
    return times_ms, sizes


def summarize(name, file_name, graph, optimal, runs, times_ms, sizes) -> dict:
    """
    Turn raw run measurements into one CSV row.

    Input: algorithm name, graph file name, the Graph, its optimal cover size
           (or None), number of runs, run times (ms) and cover sizes.
    Output: a dict keyed by CSV_COLUMNS.
    """
    size_mean = statistics.mean(sizes)
    quartiles = (statistics.quantiles(times_ms, n=4) if len(times_ms) > 1
                 else [times_ms[0]] * 3)
    return {
        "graph": os.path.splitext(file_name)[0],
        "vertices": graph.num_vertices,
        "edges": graph.num_edges,
        "optimal": optimal if optimal is not None else "",
        "algorithm": name,
        "runs": runs,
        # Median and quartiles: one slow run (background activity) barely
        # moves them, unlike the mean.
        "time_median_ms": round(statistics.median(times_ms), 3),
        "time_q1_ms": round(quartiles[0], 3),
        "time_q3_ms": round(quartiles[2], 3),
        "size_mean": round(size_mean, 2),
        "size_min": min(sizes),
        "size_max": max(sizes),
        "ratio_mean": round(size_mean / optimal, 4) if optimal else "",
    }


def benchmark_vertex_cover(data_dir: str, runs: int) -> list:
    """
    Benchmark every vertex cover algorithm on every graph file in data_dir.

    Input: folder of graph files, and how many times to run each algorithm.
    Output: a list of CSV rows (dicts), ordered by graph size.
    """
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(GRAPH_EXTENSIONS))
    if not files:
        raise SystemExit(f"No graph files found in '{data_dir}'.")

    # Warm up the interpreter and CPU on the first graph before any timing,
    # so the first (smallest) graph isn't measured on a "cold" machine.
    warmup_graph = Graph.from_file(os.path.join(data_dir, files[0]))
    for algorithm in VERTEX_COVER_ALGORITHMS.values():
        for seed in range(5):
            algorithm(warmup_graph, seed)

    rows = []
    for file_name in files:
        graph = Graph.from_file(os.path.join(data_dir, file_name))
        optimal = optimal_cover_size(file_name)
        print(f"{file_name}: {graph.num_vertices} vertices, {graph.num_edges} edges")
        for name, algorithm in VERTEX_COVER_ALGORITHMS.items():
            times_ms, sizes = time_algorithm(algorithm, graph, runs)
            row = summarize(name, file_name, graph, optimal, runs, times_ms, sizes)
            rows.append(row)
            print(f"  {name:<11} size {row['size_mean']:>8}  (opt {row['optimal']})"
                  f"  median time {row['time_median_ms']:>9.3f} ms")
    rows.sort(key=lambda r: (r["vertices"] + r["edges"], r["algorithm"]))
    return rows


def write_csv(rows: list, path: str) -> None:
    """
    Save benchmark rows to a CSV file.

    Input: list of row dicts and the output path.
    Output: none; the file is written (its folder is created if needed).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse command-line options, run the benchmark, and save the CSV."""
    parser = argparse.ArgumentParser(description="Benchmark vertex cover algorithms.")
    parser.add_argument("--data", default="data", help="folder of graph files")
    parser.add_argument("--out", default="results", help="folder for the results CSV")
    parser.add_argument("--runs", type=int, default=30, help="runs per algorithm per graph")
    args = parser.parse_args()

    rows = benchmark_vertex_cover(args.data, args.runs)
    out_path = os.path.join(args.out, "vertex_cover_results.csv")
    write_csv(rows, out_path)
    print(f"\nSaved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()