"""Make the report's plots and summary tables from the benchmark CSVs.

Reads the CSVs written by benchmark.py from the results folder and saves,
in results/plots/, each figure as both .pdf (for Overleaf) and .png.

Vertex cover (from vertex_cover_results.csv):
  - vc_runtime              median running time vs. graph size |V| + |E|
  - vc_runtime_per_edge     running time / (|V| + |E|)  (flat = linear time)
  - vc_approx_ratio         cover size / optimal cover size per graph
Graph coloring (from coloring_results.csv):
  - color_runtime           median running time vs. graph size |V| + |E|
  - color_runtime_per_edge  running time / (|V| + |E|)  (flat = linear time)
  - color_count             colors used per graph, with the lower bound K
It also writes vertex_cover_table.tex and coloring_table.tex for the report.
A problem whose CSV is missing is skipped.


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

TEXT_COLOR = "#3d3d3a"
MUTED_COLOR = "#8a8a85"
GRID_COLOR = "#e4e3df"

# Per-problem settings. Each algorithm keeps one fixed colour and marker in
# every figure, so a series is recognisable across plots.
PLOT_CONFIG = {
    "vertex_cover": {
        "prefix": "vc",
        "title": "Vertex cover",
        "value": "size",
        "reference": "optimal",
        "styles": {
            "basic": {"label": "Basic (VC2)", "color": "#2a78d6", "marker": "o"},
            "randomized": {"label": "Randomized (Pitt's U)", "color": "#eb6834",
                           "marker": "s"},
        },
        "table": {
            "caption": "Vertex cover results: mean cover size (randomized min--max "
                       "in brackets) and median running time over all runs.",
            "label": "tab:vertex-cover",
            "reference_header": "OPT",
            "value_header": "Cover size",
            "ratio_header": "Size / OPT",
        },
    },
    "coloring": {
        "prefix": "color",
        "title": "Graph coloring",
        "value": "colors",
        "reference": "lower_bound",
        "styles": {
            "natural": {"label": "Greedy, natural order", "color": "#2a78d6",
                        "marker": "o"},
            "random": {"label": "Greedy, random order", "color": "#eb6834",
                       "marker": "s"},
            "smallest_last": {"label": "Greedy, Smallest Last", "color": "#1baf7a",
                              "marker": "^"},
        },
        "table": {
            "caption": "Graph coloring results: mean colors used (random-order "
                       "min--max in brackets), the clique lower bound $K$, and "
                       "median running time over all runs.",
            "label": "tab:coloring",
            "reference_header": "$K$",
            "value_header": "Colors",
            "ratio_header": "Colors / $K$",
        },
    },
}

NUMERIC_FIELDS = {"vertices", "edges", "optimal", "lower_bound", "runs",
                  "time_median_ms", "time_q1_ms", "time_q3_ms", "ratio_mean",
                  "size_mean", "size_min", "size_max",
                  "colors_mean", "colors_min", "colors_max"}


def read_results(path: str) -> list:
    """
    Load benchmark rows from a CSV.

    Input: path to a *_results.csv file.
    Output: list of dicts with numeric fields converted to numbers
            (empty cells become None).
    """
    rows = []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            for key in NUMERIC_FIELDS & row.keys():
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


def style_category_axis(ax, graph_names: list) -> None:
    """
    Set up an x-axis of graph names (one category per graph).

    Input: a matplotlib Axes and the graph names, in plotting order.
    Output: none; ticks, labels and limits are set in place.
    """
    ax.grid(False, axis="x")  # graphs are categories; only value gridlines help
    ax.set_xlim(-0.5, len(graph_names) - 0.5)
    ax.set_xticks(range(len(graph_names)))
    ax.set_xticklabels(graph_names, rotation=30, ha="right")


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


def plot_runtime(rows: list, config: dict, plots_dir: str) -> None:
    """
    Line chart of median running time against |V| + |E| (thousands), with
    the interquartile range (25th-75th percentile) as error bars.

    Input: one problem's rows, its PLOT_CONFIG entry, and the plots folder.
    Output: none; <prefix>_runtime.pdf/.png are written.
    """
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, style in config["styles"].items():
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
    style_axes(ax, f"{config['title']} running time vs. graph size",
               "|V| + |E| (thousands)", "Median running time (ms)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_COLOR)
    save_figure(fig, plots_dir, f"{config['prefix']}_runtime")


def plot_runtime_per_edge(rows: list, config: dict, plots_dir: str) -> None:
    """
    Running time divided by |V| + |E| for each graph. A roughly flat line
    is the empirical sign of O(V + E) time.

    Input: one problem's rows, its PLOT_CONFIG entry, and the plots folder.
    Output: none; <prefix>_runtime_per_edge.pdf/.png are written.
    """
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, style in config["styles"].items():
        data = rows_for(rows, name)
        if not data:
            continue
        x = [(r["vertices"] + r["edges"]) / 1000 for r in data]
        y = [r["time_median_ms"] * 1e6 / (r["vertices"] + r["edges"]) for r in data]
        ax.plot(x, y, color=style["color"], marker=style["marker"], markersize=5,
                linewidth=2, markeredgecolor="white", markeredgewidth=1,
                label=style["label"])
    style_axes(ax, f"{config['title']}: time per element (flat line = linear time)",
               "|V| + |E| (thousands)", "Running time / (|V| + |E|)  (ns)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_COLOR)
    save_figure(fig, plots_dir, f"{config['prefix']}_runtime_per_edge")


def plot_quality_points(ax, rows: list, config: dict, transform) -> list:
    """
    Draw each algorithm's mean result per graph as a point, side by side,
    with a bar from its best to its worst run when runs differ.

    Input: an Axes, one problem's rows, its PLOT_CONFIG entry, and a function
           (row, value) -> y that turns a raw value into the plotted number.
    Output: the graph names, in plotting order.
    """
    styles = config["styles"]
    value = config["value"]
    first = next(iter(styles))
    graph_names = [r["graph"] for r in rows_for(rows, first)]
    spacing = 0.24
    offsets = {name: (i - (len(styles) - 1) / 2) * spacing for i, name in enumerate(styles)}
    for name, style in styles.items():
        data = rows_for(rows, name)
        if not data:
            continue
        x = [i + offsets[name] for i in range(len(data))]
        y = [transform(r, r[f"{value}_mean"]) for r in data]
        low = [yi - transform(r, r[f"{value}_min"]) for yi, r in zip(y, data)]
        high = [transform(r, r[f"{value}_max"]) - yi for yi, r in zip(y, data)]
        spread = [low, high] if any(low) or any(high) else None  # deterministic: none
        ax.errorbar(x, y, yerr=spread, fmt=style["marker"], color=style["color"],
                    markersize=6, capsize=3, elinewidth=1,
                    markeredgecolor="white", markeredgewidth=1, label=style["label"])
    return graph_names


def plot_approx_ratio(rows: list, config: dict, plots_dir: str) -> None:
    """
    Vertex cover: cover size divided by the known optimum for each graph.

    Input: the vertex cover rows, its PLOT_CONFIG entry, and the plots folder.
    Output: none; vc_approx_ratio.pdf/.png are written (skipped when no graph
            has a known optimum).
    """
    rows = [r for r in rows if r["optimal"]]
    if not rows:
        print("No known optimum in the results; skipping vc_approx_ratio.")
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    graph_names = plot_quality_points(ax, rows, config,
                                      lambda r, v: v / r["optimal"])
    ax.axhline(1.0, color=MUTED_COLOR, linewidth=1, linestyle="--")
    ax.text(-0.4, 1.0, "optimal", ha="left", va="bottom", fontsize=8, color=MUTED_COLOR)
    style_axes(ax, "Cover size relative to the optimum (lower is better)",
               "Graph", "Cover size / optimal size")
    style_category_axis(ax, graph_names)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_COLOR, loc="upper right")
    save_figure(fig, plots_dir, "vc_approx_ratio")


def plot_color_count(rows: list, config: dict, plots_dir: str) -> None:
    """
    Graph coloring: colors used per graph, with the clique lower bound K
    drawn as a short grey line under each graph.

    Input: the coloring rows, its PLOT_CONFIG entry, and the plots folder.
    Output: none; color_count.pdf/.png are written.
    """
    fig, ax = plt.subplots(figsize=(6, 3.8))
    graph_names = plot_quality_points(ax, rows, config, lambda r, v: v)
    bounds = {r["graph"]: r["lower_bound"] for r in rows if r["lower_bound"]}
    for i, graph in enumerate(graph_names):
        if graph in bounds:
            ax.hlines(bounds[graph], i - 0.4, i + 0.4, color=MUTED_COLOR,
                      linewidth=2, linestyle="--",
                      label="Lower bound K (clique size)" if i == 0 else None)
    style_axes(ax, "Colors used per graph (lower is better)", "Graph", "Number of colors")
    style_category_axis(ax, graph_names)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_COLOR, loc="upper left")
    save_figure(fig, plots_dir, "color_count")


def write_latex_table(rows: list, config: dict, path: str) -> None:
    """
    Write a booktabs LaTeX table summarising every graph and algorithm.

    Input: one problem's rows, its PLOT_CONFIG entry, and the output .tex path.
    Output: none; the .tex file is written (needs \\usepackage{booktabs}).
    """
    table, value, reference = config["table"], config["value"], config["reference"]
    lines = [
        r"\begin{table}[t]",
        f"  \\caption{{{table['caption']}}}",
        f"  \\label{{{table['label']}}}",
        r"  \small",
        r"  \begin{tabular}{lrrrlrrr}",
        r"    \toprule",
        f"    Graph & $|V|$ & $|E|$ & {table['reference_header']} & Algorithm & "
        f"{table['value_header']} & {table['ratio_header']} & Median time (ms) \\\\",
        r"    \midrule",
    ]
    for row in sorted(rows, key=lambda r: (r["vertices"] + r["edges"], r["algorithm"])):
        result = f"{row[f'{value}_mean']:.1f}"
        if row[f"{value}_min"] != row[f"{value}_max"]:
            result += f" [{row[f'{value}_min']:.0f}--{row[f'{value}_max']:.0f}]"
        ref = f"{row[reference]:.0f}" if row[reference] else "--"
        ratio = f"{row['ratio_mean']:.3f}" if row["ratio_mean"] else "--"
        algorithm = row["algorithm"].replace("_", " ")
        lines.append(
            f"    {row['graph']} & {row['vertices']:.0f} & {row['edges']:.0f} & {ref} & "
            f"{algorithm} & {result} & {ratio} & {row['time_median_ms']:.2f} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")
    print(f"Saved {os.path.basename(path)}")


QUALITY_PLOTS = {"vertex_cover": plot_approx_ratio, "coloring": plot_color_count}


def main() -> None:
    """Parse command-line options, then make every plot and table."""
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument("--results", default="results", help="folder holding the CSVs")
    args = parser.parse_args()
    plots_dir = os.path.join(args.results, "plots")

    found_any = False
    for problem_name, config in PLOT_CONFIG.items():
        csv_path = os.path.join(args.results, f"{problem_name}_results.csv")
        if not os.path.exists(csv_path):
            print(f"'{csv_path}' not found; skipping {problem_name}.")
            continue
        found_any = True
        rows = read_results(csv_path)
        plot_runtime(rows, config, plots_dir)
        plot_runtime_per_edge(rows, config, plots_dir)
        QUALITY_PLOTS[problem_name](rows, config, plots_dir)
        write_latex_table(rows, config,
                          os.path.join(args.results, f"{problem_name}_table.tex"))
    if not found_any:
        raise SystemExit("No results CSVs found. Run benchmark.py first.")


if __name__ == "__main__":
    main()