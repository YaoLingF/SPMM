#!/usr/bin/env python3
"""Train default PyG ClusterLoader + GCN for full-epoch comparisons."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import train_graphsage_sampled as data_mod  # noqa: E402
from profile_cluster_gcn_spmm import DEFAULT_CLUSTER_PARTS  # noqa: E402


class ClusterGCN(nn.Module):
    """Fixed GCN stack backed by PyG GCNConv."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full ClusterGCN training")
    parser.add_argument(
        "--dataset",
        type=data_mod.normalize_dataset_name,
        choices=data_mod.SUPPORTED_DATASETS,
        required=True,
    )
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--cluster-parts", type=int, default=0)
    parser.add_argument("--cluster-batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output-dir", default="results/clustergcn_sampled_train")
    parser.add_argument(
        "--directed-ogbn-arxiv",
        action="store_true",
        help="keep ogbn-arxiv directed; default converts it to undirected",
    )
    return parser.parse_args()


def make_loader_args(args: argparse.Namespace) -> SimpleNamespace:
    cluster_parts = args.cluster_parts
    if cluster_parts <= 0:
        cluster_parts = DEFAULT_CLUSTER_PARTS.get(args.dataset, 1500)
    return SimpleNamespace(
        loader="cluster",
        data_root=args.data_root,
        num_layers=args.num_layers,
        fanout="",
        batch_size=0,
        num_workers=args.num_workers,
        cluster_parts=cluster_parts,
        cluster_batch_size=args.cluster_batch_size,
    )


def train_batch(model, optimizer, batch, args: argparse.Namespace, device: torch.device):
    batch = batch.to(device, non_blocking=True)
    mask = data_mod.batch_train_mask(batch, device)
    if int(mask.sum()) == 0:
        return None, 0

    optimizer.zero_grad(set_to_none=True)
    out = model(batch.x, batch.edge_index)

    target = batch.y[mask]
    loss = data_mod.classification_loss(out[mask], target)
    loss.backward()
    optimizer.step()
    return float(loss.detach()), int(mask.sum())


def train_epoch(model, optimizer, loader, args: argparse.Namespace, device: torch.device, epoch: int) -> dict:
    model.train()
    start_wall = time.perf_counter()
    use_cuda_events = device.type == "cuda"
    if use_cuda_events:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    total_loss = 0.0
    total_examples = 0
    used_steps = 0

    for step, batch in enumerate(loader, start=1):
        if args.max_steps_per_epoch > 0 and step > args.max_steps_per_epoch:
            break

        loss_value, examples = train_batch(model, optimizer, batch, args, device)
        if loss_value is None:
            continue

        used_steps += 1
        total_examples += examples
        total_loss += loss_value * examples
        if args.log_every > 0 and used_steps % args.log_every == 0:
            print(
                f"epoch={epoch:03d} step={used_steps:04d} "
                f"examples={total_examples} "
                f"loss={total_loss / max(total_examples, 1):.4f}"
            )

    if use_cuda_events:
        end_event.record()
        torch.cuda.synchronize()
        cuda_ms = start_event.elapsed_time(end_event)
    else:
        cuda_ms = 0.0

    return {
        "epoch": epoch,
        "steps": used_steps,
        "examples": total_examples,
        "loss": total_loss / max(total_examples, 1),
        "wall_ms": (time.perf_counter() - start_wall) * 1000.0,
        "cuda_ms": cuda_ms,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    data_mod.require_module("torch_geometric")
    data_mod.set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    bundle = data_mod.load_dataset(
        args.dataset,
        Path(args.data_root),
        directed_ogbn_arxiv=args.directed_ogbn_arxiv,
    )
    largs = make_loader_args(args)
    loader = data_mod.make_loader(bundle, largs)
    model = ClusterGCN(
        in_channels=int(bundle.data.num_features),
        hidden_channels=args.hidden_channels,
        out_channels=bundle.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(
        f"dataset={args.dataset} loader=cluster variant=default "
        f"nodes={bundle.data.num_nodes} edges={bundle.data.num_edges} "
        f"features={bundle.data.num_features} classes={bundle.num_classes} "
        f"task={bundle.task_type}"
    )
    print(
        f"ClusterGCN layers={args.num_layers} hidden={args.hidden_channels} "
        f"dropout={args.dropout} cluster_parts={largs.cluster_parts} "
        f"cluster_batch={args.cluster_batch_size}"
    )

    rows = []
    for epoch in range(1, args.epochs + 1):
        row = train_epoch(model, optimizer, loader, args, device, epoch)
        row.update(
            {
                "dataset": args.dataset,
                "loader": "cluster",
                "variant": "default",
                "model": "gcn",
                "num_layers": args.num_layers,
                "hidden_channels": args.hidden_channels,
                "cluster_parts": largs.cluster_parts,
                "cluster_batch_size": args.cluster_batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }
        )
        rows.append(row)
        print(
            f"epoch={epoch:03d} steps={row['steps']} examples={row['examples']} "
            f"loss={row['loss']:.4f} wall={row['wall_ms']:.2f}ms "
            f"cuda={row['cuda_ms']:.2f}ms"
        )

    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"{args.dataset}_cluster_default_train.csv"
    write_rows(csv_path, rows)
    print(f"log_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
