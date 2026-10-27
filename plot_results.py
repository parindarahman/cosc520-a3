"""Make the report's plots and summary table from the benchmark CSV.

Reads results/vertex_cover_results.csv (written by benchmark.py) and saves,
in results/plots/, each figure as both .pdf (for Overleaf) and .png:
  - vc_runtime          median running time vs. graph size |V| + |E|
  - vc_runtime_per_edge running time divided by |V| + |E| (flat = linear time)
  - vc_approx_ratio     cover size / optimal cover size for each graph
It also writes results/vertex_cover_table.tex, a LaTeX table for the report.

Usage:
    python plot_results.py
    python plot_results.py --results results
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")  # draw to files only; no window needed
import matplotlib.pyplot as plt  # noqa: E402

# One fixed colour per algorithm, so a series keeps its colour in every figure.
ALGORITHM_STYLE = {
    "basic": {"label": "Basic (VC2)", "color": "#2a78d6", "marker": "o"},
    "randomized": {"label": "Randomized (Pitt's U)", "color": "#eb6834", "marker": "s"},
}
TEXT_COLOR = "#3d3d3a"
MUTED_COLOR = "#8a8a85"
GRID_COLOR = "#e4e3df"


def read_results(path: str) -> list:
    """
    Load benchmark rows from the CSV.

    Input: path to vertex_cover_results.csv.
    Output: list of dicts with numeric fields converted to numbers.
    """
    numeric = {"vertices", "edges", "optimal", "runs", "time_median_ms", "time_q1_ms", "time_q3_ms",
               "size_mean", "size_min", "size_max", "ratio_mean"}
    rows = []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            for key in numeric:
                row[key] = float(row[key]) if row[key] != "" else None
            rows.append(row)
    return rows


def rows_for(rows: list, algorithm: str) -> list:
    """
    Input: all rows and an algorithm name.
    Output: that algorithm's rows, sorted by graph size |V| + |E|.
    """
    selected = [r for r in rows if r["algorithm"] == algorithm]
    return sorted(selected, key=lambda r: r["vertices"] + r["edges"])


def style_axes(ax, title: str, x_label: str, y_label: str) -> None:
    """
    Apply the shared look: recessive grid and axes, no top/right frame.

    Input: a matplotlib Axes and its title and axis labels.
    Output: none; the axes are styled in place.
    """
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_COLOR, pad=10)
    ax.set_xlabel(x_label, fontsize=9, color=TEXT_COLOR)
    ax.set_ylabel(y_label, fontsize=9, color=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelcolor=TEXT_COLOR, labelsize=8)


def save_figure(fig, plots_dir: str, name: str) -> None:
    """
    Save a figure as .pdf and .png, then close it.

    Input: the figure, output folder and base file name.
    Output: none; two files are written.
    """
    os.makedirs(plots_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(plots_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.pdf / .png")


def plot_runtime(rows: list, plots_dir: str) -> None:
    """
    Line chart of median running time against |V| + |E| (thousands),
    with the interquartile range (25th-75th percentile) as error bars.

    Input: all benchmark rows and the plots folder.
    Output: none; vc_runtime.pdf/.png are written.
    """
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, style in ALGORITHM_STYLE.items():
        data = rows_for(rows, name)
        if not data:
            continue
        x = [(r["vertices"] + r["edges"]) / 1000 for r in data]
        y = [r["time_median_ms"] for r in data]
        err = [[r["time_median_ms"] - r["time_q1_ms"] for r in data],
               [r["time_q3_ms"] - r["time_median_ms"] for r in data]]
        ax.errorbar(x, y, yerr=err, color=style["color"], marker=style["marker"],
                    markersize=5, linewidth=2, capsize=3, elinewidth=1,
                    markeredgecolor="white", markeredgewidth=1, label=style["label"])
    style_axes(ax, "Vertex cover running time vs. graph size",
               "|V| + |E| (thousands)", "Median running time (ms)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_COLOR)
    save_figure(fig, plots_dir, "vc_runtime")


def plot_runtime_per_edge(rows: list, plots_dir: str) -> None:
    """
    Running time divided by |V| + |E| for each graph. A roughly flat line
    is the empirical sign of O(V + E) time.

    Input: all benchmark rows and the plots folder.
    Output: none; vc_runtime_per_edge.pdf/.png are written.
    """
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, style in ALGORITHM_STYLE.items():
        data = rows_for(rows, name)
        if not data:
            continue
        x = [(r["vertices"] + r["edges"]) / 1000 for r in data]
        y = [r["time_median_ms"] * 1e6 / (r["vertices"] + r["edges"]) for r in data]
        ax.plot(x, y, color=style["color"], marker=style["marker"], markersize=5,
                linewidth=2, markeredgecolor="white", markeredgewidth=1,
                label=style["label"])
    style_axes(ax, "Time per element (flat line = linear time)",
               "|V| + |E| (thousands)", "Running time / (|V| + |E|)  (ns)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_COLOR)
    save_figure(fig, plots_dir, "vc_runtime_per_edge")


def plot_approx_ratio(rows: list, plots_dir: str) -> None:
    """
    Cover size divided by the known optimum for each graph. The randomized
    series shows its mean, with a bar from its best to its worst run.

    Input: all benchmark rows and the plots folder.
    Output: none; vc_approx_ratio.pdf/.png are written (skipped when no
            graph has a known optimum).
    """
    if not any(r["optimal"] for r in rows):
        print("No known optimum in the results; skipping vc_approx_ratio.")
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    graph_names = [r["graph"] for r in rows_for(rows, "basic") if r["optimal"]]
    positions = range(len(graph_names))
    offsets = {"basic": -0.12, "randomized": 0.12}
    for name, style in ALGORITHM_STYLE.items():
        data = [r for r in rows_for(rows, name) if r["optimal"]]
        if not data:
            continue
        x = [p + offsets[name] for p in positions]
        y = [r["size_mean"] / r["optimal"] for r in data]
        low = [yi - r["size_min"] / r["optimal"] for yi, r in zip(y, data)]
        high = [r["size_max"] / r["optimal"] - yi for yi, r in zip(y, data)]
        spread = [low, high] if any(low) or any(high) else None  # basic has no spread
        ax.errorbar(x, y, yerr=spread, fmt=style["marker"], color=style["color"],
                    markersize=6, capsize=3, elinewidth=1,
                    markeredgecolor="white", markeredgewidth=1, label=style["label"])
    ax.axhline(1.0, color=MUTED_COLOR, linewidth=1, linestyle="--")
    ax.text(-0.4, 1.0, "optimal", ha="left", va="bottom", fontsize=8, color=MUTED_COLOR)
    style_axes(ax, "Cover size relative to the optimum (lower is better)",
               "Graph", "Cover size / optimal size")
    ax.grid(False, axis="x")  # graphs are categories; only value gridlines help
    ax.set_xlim(-0.5, len(graph_names) - 0.5)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(graph_names, rotation=30, ha="right")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_COLOR, loc="upper right")
    save_figure(fig, plots_dir, "vc_approx_ratio")


def write_latex_table(rows: list, path: str) -> None:
    """
    Write a booktabs LaTeX table summarising every graph and algorithm.

    Input: all benchmark rows and the output .tex path.
    Output: none; the .tex file is written (needs \\usepackage{booktabs}).
    """
    lines = [
        r"\begin{table}[t]",
        r"  \caption{Vertex cover results: mean cover size (randomized min--max in brackets) and median running time over all runs.}",
        r"  \label{tab:vertex-cover}",
        r"  \small",
        r"  \begin{tabular}{lrrrlrrr}",
        r"    \toprule",
        r"    Graph & $|V|$ & $|E|$ & OPT & Algorithm & Cover size & Ratio & Median time (ms) \\",
        r"    \midrule",
    ]
    for row in sorted(rows, key=lambda r: (r["vertices"] + r["edges"], r["algorithm"])):
        size = f"{row['size_mean']:.1f}"
        if row["size_min"] != row["size_max"]:
            size += f" [{row['size_min']:.0f}--{row['size_max']:.0f}]"
        optimal = f"{row['optimal']:.0f}" if row["optimal"] else "--"
        ratio = f"{row['ratio_mean']:.3f}" if row["ratio_mean"] else "--"
        lines.append(
            f"    {row['graph']} & {row['vertices']:.0f} & {row['edges']:.0f} & {optimal} & "
            f"{row['algorithm']} & {size} & {ratio} & {row['time_median_ms']:.2f} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")
    print(f"Saved {os.path.basename(path)}")


def main() -> None:
    """Parse command-line options, then make every plot and the table."""
    parser = argparse.ArgumentParser(description="Plot vertex cover benchmark results.")
    parser.add_argument("--results", default="results", help="folder holding the CSV")
    args = parser.parse_args()

    csv_path = os.path.join(args.results, "vertex_cover_results.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"'{csv_path}' not found. Run benchmark.py first.")
    rows = read_results(csv_path)
    plots_dir = os.path.join(args.results, "plots")

    plot_runtime(rows, plots_dir)
    plot_runtime_per_edge(rows, plots_dir)
    plot_approx_ratio(rows, plots_dir)
    write_latex_table(rows, os.path.join(args.results, "vertex_cover_table.tex"))


if __name__ == "__main__":
    main()