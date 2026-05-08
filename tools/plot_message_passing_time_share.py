#!/usr/bin/env python3
"""Plot message-passing CUDA compute time share from profiler CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASET_ORDER = [
    "ogbn_arxiv",
    "ogbn_products",
    "flickr",
    "yelp",
    "amazon_products",
    "reddit",
]

DATASET_LABELS = {
    "ogbn_arxiv": "ogbn-arxiv",
    "ogbn_products": "ogbn-products",
    "flickr": "Flickr",
    "yelp": "Yelp",
    "amazon_products": "AmazonProducts",
    "reddit": "Reddit",
}

WORKLOADS = [
    ("neighbor_graphsage", "GraphSAGE", "#4C78A8"),
    ("cluster_gcn", "ClusterGCN", "#F58518"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a message-passing time-share figure for paper use"
    )
    parser.add_argument(
        "--graphsage-csv",
        default=(
            "results/gnn_final/graphsage_spmm_six_full/"
            "graphsage_neighbor_spmm_share_summary.csv"
        ),
    )
    parser.add_argument(
        "--cluster-csv",
        default=(
            "results/gnn_final/cluster_gcn_spmm_six_full/"
            "cluster_gcn_spmm_share_summary.csv"
        ),
    )
    parser.add_argument("--output-dir", default="results/gnn_final/figures")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--no-annotate", action="store_true")
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_csv(path: Path, workload: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing CSV: {path}")

    df = pd.read_csv(path)
    required = {"dataset", "message_passing_pct_raw_compute"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df[["dataset", "message_passing_pct_raw_compute"]].copy()
    df["workload"] = workload
    df["dataset_order"] = df["dataset"].map(
        {name: idx for idx, name in enumerate(DATASET_ORDER)}
    )
    df["dataset_order"] = df["dataset_order"].fillna(len(DATASET_ORDER)).astype(int)
    return df


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.concat(
        [
            load_csv(Path(args.graphsage_csv), "neighbor_graphsage"),
            load_csv(Path(args.cluster_csv), "cluster_gcn"),
        ],
        ignore_index=True,
    )
    return df.sort_values(["dataset_order", "dataset", "workload"])


def ordered_datasets(df: pd.DataFrame) -> list[str]:
    return (
        df[["dataset", "dataset_order"]]
        .drop_duplicates()
        .sort_values(["dataset_order", "dataset"])["dataset"]
        .tolist()
    )


def annotate_bars(ax: plt.Axes, bars) -> None:
    ymax = ax.get_ylim()[1]
    for bar in bars:
        value = float(bar.get_height())
        ax.annotate(
            f"{value:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )
    ax.set_ylim(top=ymax)


def plot_message_passing(df: pd.DataFrame, output_dir: Path, dpi: int, annotate: bool) -> None:
    datasets = ordered_datasets(df)
    labels = [DATASET_LABELS.get(dataset, dataset) for dataset in datasets]
    x = list(range(len(datasets)))
    width = 0.36
    metric = "message_passing_pct_raw_compute"

    fig, ax = plt.subplots(figsize=(5.8, 2.45))
    max_value = 0.0
    for idx, (workload, label, color) in enumerate(WORKLOADS):
        part = df[df["workload"] == workload].set_index("dataset")
        values = [float(part.loc[dataset, metric]) for dataset in datasets]
        max_value = max(max_value, max(values))
        offset = (idx - 0.5) * width
        bars = ax.bar(
            [pos + offset for pos in x],
            values,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.35,
        )
        if annotate:
            annotate_bars(ax, bars)

    ax.set_ylabel("Time share (%)")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", rotation=24)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
        tick.set_rotation_mode("anchor")
    ax.set_ylim(0, max_value * (1.22 if annotate else 1.12))
    ax.grid(axis="y", alpha=0.28)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "message_passing_time_share.pdf", bbox_inches="tight")
    fig.savefig(
        output_dir / "message_passing_time_share.png",
        bbox_inches="tight",
        dpi=dpi,
    )
    plt.close(fig)


def write_merged_csv(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "message_passing_time_share.csv"
    df[["dataset", "workload", "message_passing_pct_raw_compute"]].to_csv(
        path,
        index=False,
    )
    return path


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    output_dir = Path(args.output_dir)
    df = load_data(args)
    csv_path = write_merged_csv(df, output_dir)
    plot_message_passing(
        df,
        output_dir,
        dpi=args.dpi,
        annotate=not args.no_annotate,
    )

    print(f"csv={csv_path}")
    print(output_dir / "message_passing_time_share.pdf")
    print(output_dir / "message_passing_time_share.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
