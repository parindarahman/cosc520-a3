"""Benchmark the vertex cover and graph coloring algorithms on every graph
in a data folder.

For each graph file it runs every algorithm several times, checks that each
result is valid, and records running time and solution quality (cover size,
or number of colors). Results go to one CSV per problem, which
plot_results.py turns into plots and tables.


Usage:
    python benchmark.py                          # both problems, 30 runs
    python benchmark.py --problem coloring       # just one problem
    python benchmark.py --runs 10 --data data --out results
"""

import argparse
import csv
import os
import random
import re
import statistics
import time

from coloring import (
    basic_greedy_coloring,
    count_colors,
    is_valid_coloring,
    random_order_coloring,
    smallest_last_coloring,
)
from graph import Graph
from vertex_cover import basic_vertex_cover, is_vertex_cover, randomized_vertex_cover

GRAPH_EXTENSIONS = (".mtx", ".clq", ".col", ".mis")


def frb_parameters(file_name: str):
    """
    Read N (number of groups) and K (group size) from an FRB file name.

    Input: a file name such as 'frb30-15-1.mtx'.
    Output: (N, K), or None if the name is not an FRB name.
    """
    match = re.match(r"frb(\d+)-(\d+)", file_name)
    return (int(match.group(1)), int(match.group(2))) if match else None


def optimal_cover_size(file_name: str):
    """
    Known optimum vertex cover for an FRB instance.

    frbN-K has N*K vertices and a maximum independent set of size N,
    so the minimum vertex cover has N*K - N vertices.

    Input: an FRB file name.
    Output: the optimal cover size, or None if the name is not an FRB name.
    """
    params = frb_parameters(file_name)
    return params[0] * params[1] - params[0] if params else None


def color_lower_bound(file_name: str):
    """
    Lower bound on the number of colors for an FRB instance.

    Each of the N groups in frbN-K is a clique of K vertices, and a clique
    of K vertices needs K different colors, so any coloring uses >= K.

    Input: an FRB file name.
    Output: K, or None if the name is not an FRB name.
    """
    params = frb_parameters(file_name)
    return params[1] if params else None


# Each problem lists its algorithms (each takes (graph, seed); deterministic
# ones ignore the seed), how to check and score a result, and what the
# reference value for a graph is (the optimum, or a lower bound).
PROBLEMS = {
    "vertex_cover": {
        "algorithms": {
            "basic": lambda g, seed: basic_vertex_cover(g),
            "randomized": lambda g, seed: randomized_vertex_cover(g, random.Random(seed)),
        },
        "is_valid": is_vertex_cover,
        "score": len,                      # cover size
        "reference": optimal_cover_size,
        "value_name": "size",
        "reference_name": "optimal",
    },
    "coloring": {
        "algorithms": {
            "natural": lambda g, seed: basic_greedy_coloring(g),
            "random": lambda g, seed: random_order_coloring(g, random.Random(seed)),
            "smallest_last": lambda g, seed: smallest_last_coloring(g),
        },
        "is_valid": is_valid_coloring,
        "score": count_colors,             # number of colors
        "reference": color_lower_bound,
        "value_name": "colors",
        "reference_name": "lower_bound",
    },
}


def csv_columns(problem: dict) -> list:
    """
    Input: a PROBLEMS entry.
    Output: the CSV column names for that problem.
    """
    value, reference = problem["value_name"], problem["reference_name"]
    return ["graph", "vertices", "edges", reference, "algorithm", "runs",
            "time_median_ms", "time_q1_ms", "time_q3_ms",
            f"{value}_mean", f"{value}_min", f"{value}_max", "ratio_mean"]


def time_algorithm(algorithm, graph: Graph, runs: int, is_valid, score):
    """
    Run one algorithm several times, checking and scoring each result.

    Input: an algorithm taking (graph, seed), a Graph, the number of runs,
           a validity check (graph, result) -> bool, and a score function
           result -> number. Run i uses seed i, so results are reproducible.
    Output: (list of run times in milliseconds, list of scores).
    Raises: AssertionError if any run returns an invalid result.
    """
    times_ms, scores = [], []
    algorithm(graph, 0)  # warm-up run, not timed
    for seed in range(runs):
        start = time.perf_counter()
        result = algorithm(graph, seed)
        times_ms.append((time.perf_counter() - start) * 1000)
        assert is_valid(graph, result), "algorithm returned an invalid result"
        scores.append(score(result))
    return times_ms, scores


def summarize(problem, name, file_name, graph, runs, times_ms, scores) -> dict:
    """
    Turn raw run measurements into one CSV row.

    Input: the PROBLEMS entry, algorithm name, graph file name, the Graph,
           number of runs, run times (ms) and scores.
    Output: a dict keyed by csv_columns(problem).
    """
    value, reference_name = problem["value_name"], problem["reference_name"]
    reference = problem["reference"](file_name)
    mean_score = statistics.mean(scores)
    quartiles = (statistics.quantiles(times_ms, n=4) if len(times_ms) > 1
                 else [times_ms[0]] * 3)
    return {
        "graph": os.path.splitext(file_name)[0],
        "vertices": graph.num_vertices,
        "edges": graph.num_edges,
        reference_name: reference if reference is not None else "",
        "algorithm": name,
        "runs": runs,
        # Median and quartiles: one slow run (background activity) barely
        # moves them, unlike the mean.
        "time_median_ms": round(statistics.median(times_ms), 3),
        "time_q1_ms": round(quartiles[0], 3),
        "time_q3_ms": round(quartiles[2], 3),
        f"{value}_mean": round(mean_score, 2),
        f"{value}_min": min(scores),
        f"{value}_max": max(scores),
        "ratio_mean": round(mean_score / reference, 4) if reference else "",
    }


def run_benchmark(problem: dict, graphs: list, runs: int) -> list:
    """
    Benchmark every algorithm of one problem on every loaded graph.

    Input: a PROBLEMS entry, a list of (file name, Graph) pairs, and the
           number of runs per algorithm.
    Output: a list of CSV rows (dicts), ordered by graph size.
    """
    algorithms = problem["algorithms"]
    # Warm up the interpreter and CPU before any timing, so the first
    # (smallest) graph isn't measured on a "cold" machine.
    _, warmup_graph = graphs[0]
    for algorithm in algorithms.values():
        for seed in range(5):
            algorithm(warmup_graph, seed)

    rows = []
    value = problem["value_name"]
    for file_name, graph in graphs:
        print(f"{file_name}: {graph.num_vertices} vertices, {graph.num_edges} edges")
        for name, algorithm in algorithms.items():
            times_ms, scores = time_algorithm(
                algorithm, graph, runs, problem["is_valid"], problem["score"])
            row = summarize(problem, name, file_name, graph, runs, times_ms, scores)
            rows.append(row)
            print(f"  {name:<14} {value} {row[f'{value}_mean']:>8}"
                  f"  ({problem['reference_name']} {row[problem['reference_name']]})"
                  f"  median time {row['time_median_ms']:>9.3f} ms")
    rows.sort(key=lambda r: (r["vertices"] + r["edges"], r["algorithm"]))
    return rows


def load_graphs(data_dir: str) -> list:
    """
    Load every graph file in a folder, smallest file first.

    Input: folder of graph files.
    Output: a list of (file name, Graph) pairs.
    """
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(GRAPH_EXTENSIONS)]
    if not files:
        raise SystemExit(f"No graph files found in '{data_dir}'.")
    files.sort(key=lambda f: os.path.getsize(os.path.join(data_dir, f)))
    graphs = []
    for file_name in files:
        print(f"Loading {file_name} ...")
        graphs.append((file_name, Graph.from_file(os.path.join(data_dir, file_name))))
    return graphs


def write_csv(rows: list, columns: list, path: str) -> None:
    """
    Save benchmark rows to a CSV file.

    Input: list of row dicts, the column names, and the output path.
    Output: none; the file is written (its folder is created if needed).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse command-line options, run the benchmarks, and save the CSVs."""
    parser = argparse.ArgumentParser(description="Benchmark vertex cover and coloring.")
    parser.add_argument("--data", default="data", help="folder of graph files")
    parser.add_argument("--out", default="results", help="folder for the results CSVs")
    parser.add_argument("--runs", type=int, default=30, help="runs per algorithm per graph")
    parser.add_argument("--problem", choices=["all", *PROBLEMS], default="all",
                        help="which problem to benchmark")
    args = parser.parse_args()

    graphs = load_graphs(args.data)  # load once, reuse for every problem
    chosen = PROBLEMS if args.problem == "all" else {args.problem: PROBLEMS[args.problem]}
    for problem_name, problem in chosen.items():
        print(f"\n=== {problem_name} ===")
        rows = run_benchmark(problem, graphs, args.runs)
        out_path = os.path.join(args.out, f"{problem_name}_results.csv")
        write_csv(rows, csv_columns(problem), out_path)
        print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()