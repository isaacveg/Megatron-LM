#!/usr/bin/env python3

import argparse
import csv
import os
from collections import OrderedDict

import matplotlib.pyplot as plt


PLOT_SPECS = [
    {
        "title": "Train Loss",
        "phase": "train",
        "split": "train",
        "eval_scope": "",
        "metric": "loss",
        "ylabel": "Loss",
    },
    {
        "title": "Validation Loss",
        "phase": "eval",
        "split": "validation",
        "eval_scope": "periodic_validation",
        "metric": "loss",
        "ylabel": "Loss",
    },
    {
        "title": "Validation PPL",
        "phase": "eval",
        "split": "validation",
        "eval_scope": "periodic_validation",
        "metric": "ppl",
        "ylabel": "PPL",
    },
]

COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
]
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "P", "X"]


def read_rows(path):
    with open(path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["series_rank"] = int(row["series_rank"])
        row["iteration"] = int(row["iteration"])
        row["value"] = float(row["value"])
    return rows


def ordered_series(rows):
    series = OrderedDict()
    for row in sorted(rows, key=lambda item: item["series_rank"]):
        series.setdefault(row["label"], row["series_rank"])
    return list(series.keys())


def filter_rows(rows, spec):
    filtered = []
    for row in rows:
        if row["phase"] != spec["phase"]:
            continue
        if row["split"] != spec["split"]:
            continue
        if row["metric"] != spec["metric"]:
            continue
        if row["eval_scope"] != spec["eval_scope"]:
            continue
        filtered.append(row)
    return filtered


def plot_curves(rows, output_prefix):
    labels = ordered_series(rows)
    style_map = {}
    for index, label in enumerate(labels):
        style_map[label] = {
            "color": COLORS[index % len(COLORS)],
            "linestyle": LINESTYLES[index % len(LINESTYLES)],
            "marker": MARKERS[index % len(MARKERS)],
        }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for axis, spec in zip(axes, PLOT_SPECS):
        spec_rows = filter_rows(rows, spec)
        for label in labels:
            label_rows = [row for row in spec_rows if row["label"] == label]
            if not label_rows:
                continue
            label_rows.sort(key=lambda item: item["iteration"])
            style = style_map[label]
            axis.plot(
                [row["iteration"] for row in label_rows],
                [row["value"] for row in label_rows],
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4,
                linewidth=2,
            )

        axis.set_title(spec["title"])
        axis.set_xlabel("Iteration")
        axis.set_ylabel(spec["ylabel"])
        axis.grid(True, alpha=0.25)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    png_path = f"{output_prefix}.png"
    pdf_path = f"{output_prefix}.pdf"
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment curves from a combined curve CSV.")
    parser.add_argument("input_csv", help="Combined curve CSV produced by build_curve_table.py.")
    parser.add_argument("output_prefix", help="Output path prefix, without extension.")
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    plot_curves(rows, args.output_prefix)


if __name__ == "__main__":
    main()
