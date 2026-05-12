#!/usr/bin/env python3
"""Run full training comparisons for default PyG SpMM vs Our SpMM."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import train_graphsage_sampled as data_mod


DEFAULT_DATASETS = data_mod.DEFAULT_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full training SpMM comparison")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--fanout", default="10,10")
    parser.add_argument("--cluster-parts", type=int, default=0)
    parser.add_argument("--cluster-batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--output-dir", default="results/full_training_spmm_compare")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [
        data_mod.normalize_dataset_name(item)
        for item in value.split(",")
        if item.strip()
    ]


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def maybe_run(cmd: list[str], output_csv: Path, skip_existing: bool) -> None:
    if skip_existing and output_csv.exists():
        print(f"skip existing: {output_csv}", flush=True)
        return
    run(cmd)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = parse_csv_list(args.datasets)
    py = sys.executable

    for dataset in datasets:
        common = [
            "--dataset",
            dataset,
            "--data-root",
            args.data_root,
            "--epochs",
            str(args.epochs),
            "--max-steps-per-epoch",
            "0",
            "--num-layers",
            str(args.num_layers),
            "--hidden-channels",
            str(args.hidden_channels),
            "--num-workers",
            str(args.num_workers),
            "--log-every",
            str(args.log_every),
        ]

        default_neighbor_dir = out_dir / "neighbor_default"
        default_neighbor_csv = (
            default_neighbor_dir / f"{dataset}_neighbor_graphsage_train.csv"
        )
        maybe_run(
            [
                py,
                "tools/train_graphsage_sampled.py",
                *common,
                "--loader",
                "neighbor",
                "--batch-size",
                str(args.batch_size),
                "--fanout",
                args.fanout,
                "--output-dir",
                str(default_neighbor_dir),
            ],
            default_neighbor_csv,
            args.skip_existing,
        )

        our_neighbor_dir = out_dir / "neighbor_our"
        our_neighbor_csv = our_neighbor_dir / f"{dataset}_neighbor_oursage_train.csv"
        maybe_run(
            [
                py,
                "tools/train_oursage_sampled.py",
                *common,
                "--loader",
                "neighbor",
                "--batch-size",
                str(args.batch_size),
                "--fanout",
                args.fanout,
                "--output-dir",
                str(our_neighbor_dir),
            ],
            our_neighbor_csv,
            args.skip_existing,
        )

        cluster_default_dir = out_dir / "cluster_default"
        cluster_default_csv = cluster_default_dir / f"{dataset}_cluster_default_train.csv"
        maybe_run(
            [
                py,
                "tools/train_clustergcn_sampled.py",
                *common,
                "--cluster-parts",
                str(args.cluster_parts),
                "--cluster-batch-size",
                str(args.cluster_batch_size),
                "--output-dir",
                str(cluster_default_dir),
            ],
            cluster_default_csv,
            args.skip_existing,
        )

        cluster_our_dir = out_dir / "cluster_our"
        cluster_our_csv = cluster_our_dir / f"{dataset}_cluster_our_train.csv"
        maybe_run(
            [
                py,
                "tools/train_ourgcn_sampled.py",
                *common,
                "--cluster-parts",
                str(args.cluster_parts),
                "--cluster-batch-size",
                str(args.cluster_batch_size),
                "--output-dir",
                str(cluster_our_dir),
            ],
            cluster_our_csv,
            args.skip_existing,
        )

    summary_rows = []
    for dataset in datasets:
        pairs = [
            (
                "neighbor_graphsage",
                out_dir / "neighbor_default" / f"{dataset}_neighbor_graphsage_train.csv",
                out_dir / "neighbor_our" / f"{dataset}_neighbor_oursage_train.csv",
            ),
            (
                "cluster_gcn",
                out_dir / "cluster_default" / f"{dataset}_cluster_default_train.csv",
                out_dir / "cluster_our" / f"{dataset}_cluster_our_train.csv",
            ),
        ]
        for workload, default_csv, our_csv in pairs:
            default_rows = read_rows(default_csv)
            our_rows = read_rows(our_csv)
            for default_row, our_row in zip(default_rows, our_rows):
                default_wall = float(default_row["wall_ms"])
                our_wall = float(our_row["wall_ms"])
                default_cuda = float(default_row["cuda_ms"])
                our_cuda = float(our_row["cuda_ms"])
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "workload": workload,
                        "epoch": default_row["epoch"],
                        "steps": default_row["steps"],
                        "examples": default_row["examples"],
                        "default_wall_ms": f"{default_wall:.6f}",
                        "our_wall_ms": f"{our_wall:.6f}",
                        "wall_speedup": f"{default_wall / our_wall:.6f}",
                        "default_cuda_ms": f"{default_cuda:.6f}",
                        "our_cuda_ms": f"{our_cuda:.6f}",
                        "cuda_speedup": f"{default_cuda / our_cuda:.6f}",
                        "default_loss": default_row["loss"],
                        "our_loss": our_row["loss"],
                    }
                )

            default_wall_total = sum(float(row["wall_ms"]) for row in default_rows)
            our_wall_total = sum(float(row["wall_ms"]) for row in our_rows)
            default_cuda_total = sum(float(row["cuda_ms"]) for row in default_rows)
            our_cuda_total = sum(float(row["cuda_ms"]) for row in our_rows)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "workload": workload,
                    "epoch": "total",
                    "steps": sum(int(row["steps"]) for row in default_rows),
                    "examples": sum(int(row["examples"]) for row in default_rows),
                    "default_wall_ms": f"{default_wall_total:.6f}",
                    "our_wall_ms": f"{our_wall_total:.6f}",
                    "wall_speedup": f"{default_wall_total / our_wall_total:.6f}",
                    "default_cuda_ms": f"{default_cuda_total:.6f}",
                    "our_cuda_ms": f"{our_cuda_total:.6f}",
                    "cuda_speedup": f"{default_cuda_total / our_cuda_total:.6f}",
                    "default_loss": default_rows[-1]["loss"],
                    "our_loss": our_rows[-1]["loss"],
                }
            )

    summary_csv = out_dir / "full_training_spmm_compare_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"summary_csv={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
