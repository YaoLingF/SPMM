#!/usr/bin/env python3
"""Train a fixed GraphSAGE sampled-GNN model with PyG loaders.

The model architecture is defined in this file.  Dataset, loader type, sampling
fanout, batch size, epochs, and other run parameters are provided as command
line arguments.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


SUPPORTED_DATASETS = (
    "ogbn_arxiv",
    "ogbn_products",
    "flickr",
    "yelp",
    "amazon_products",
    "reddit",
)
DEFAULT_DATASETS = SUPPORTED_DATASETS

_DATASET_ALIASES = {
    "ogbn_arxiv": "ogbn_arxiv",
    "ogbn-arxiv": "ogbn_arxiv",
    "ogbnarxiv": "ogbn_arxiv",
    "ogbn_products": "ogbn_products",
    "ogbn-products": "ogbn_products",
    "ogbnproducts": "ogbn_products",
    "flickr": "flickr",
    "yelp": "yelp",
    "amazon_products": "amazon_products",
    "amazon-products": "amazon_products",
    "amazonproducts": "amazon_products",
    "reddit": "reddit",
}


@dataclass
class DatasetBundle:
    name: str
    data: object
    num_classes: int
    train_nodes: object
    task_type: str = "multiclass"


class GraphSAGE(nn.Module):
    """Fixed sampled GraphSAGE model used by this experiment."""

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
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyG sampled GraphSAGE training")
    parser.add_argument(
        "--dataset",
        type=normalize_dataset_name,
        choices=SUPPORTED_DATASETS,
        required=True,
    )
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--loader", choices=("neighbor", "cluster"), default="neighbor")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=0,
        help="0 means use the whole loader epoch",
    )
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--fanout", default="10,10")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cluster-parts", type=int, default=1500)
    parser.add_argument("--cluster-batch-size", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output-dir", default="results/graphsage_sampled_train")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--directed-ogbn-arxiv",
        action="store_true",
        help="keep ogbn-arxiv directed; default converts it to undirected",
    )
    return parser.parse_args()


def require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"required module is not installed: {name}")


def normalize_dataset_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    canonical = _DATASET_ALIASES.get(key)
    if canonical is None:
        canonical = _DATASET_ALIASES.get(key.replace("_", ""))
    if canonical is None:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise argparse.ArgumentTypeError(
            f"unsupported dataset: {name}; supported datasets: {supported}"
        )
    return canonical


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_fanout(value: str, num_layers: int) -> list[int]:
    fanout = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(fanout) == 1 and num_layers > 1:
        fanout = fanout * num_layers
    if len(fanout) != num_layers:
        raise ValueError(
            f"fanout length ({len(fanout)}) must match num_layers ({num_layers})"
        )
    return fanout


def torch_load_ogb_compat():
    orig_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_torch_load(*args, **kwargs)

    return orig_torch_load, patched_torch_load


def is_multilabel_target(target: torch.Tensor) -> bool:
    return target.dim() > 1 and target.size(-1) > 1


def infer_task_type(data) -> str:
    y = getattr(data, "y", None)
    if torch.is_tensor(y) and is_multilabel_target(y):
        return "multilabel"
    return "multiclass"


def infer_num_classes(dataset, data, task_type: str) -> int:
    if task_type == "multilabel":
        return int(data.y.size(-1))
    return int(dataset.num_classes)


def make_dataset_bundle(
    name: str,
    dataset,
    data,
    train_nodes,
) -> DatasetBundle:
    task_type = infer_task_type(data)
    return DatasetBundle(
        name=name,
        data=data,
        num_classes=infer_num_classes(dataset, data, task_type),
        train_nodes=train_nodes,
        task_type=task_type,
    )


def classification_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if is_multilabel_target(target):
        return F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype))
    return F.cross_entropy(logits, target.view(-1).long())


def load_dataset(name: str, data_root: Path, directed_ogbn_arxiv: bool) -> DatasetBundle:
    from torch_geometric.datasets import AmazonProducts, Flickr, Reddit, Yelp
    import torch_geometric.transforms as T

    name = normalize_dataset_name(name)

    if name == "reddit":
        dataset = Reddit(root=str(data_root / "Reddit"))
        data = dataset[0]
        return make_dataset_bundle(name, dataset, data, data.train_mask)

    if name in ("flickr", "yelp", "amazon_products"):
        dataset_cls = {
            "flickr": Flickr,
            "yelp": Yelp,
            "amazon_products": AmazonProducts,
        }[name]
        root_name = {
            "flickr": "Flickr",
            "yelp": "Yelp",
            "amazon_products": "AmazonProducts",
        }[name]
        dataset = dataset_cls(root=str(data_root / root_name))
        data = dataset[0]
        return make_dataset_bundle(name, dataset, data, data.train_mask)

    if name in ("ogbn_arxiv", "ogbn_products"):
        require_module("ogb")
        from ogb.nodeproppred import PygNodePropPredDataset

        ogb_name = {
            "ogbn_arxiv": "ogbn-arxiv",
            "ogbn_products": "ogbn-products",
        }[name]
        transform = (
            T.ToUndirected()
            if name == "ogbn_arxiv" and not directed_ogbn_arxiv
            else None
        )
        orig_torch_load, patched_torch_load = torch_load_ogb_compat()
        try:
            torch.load = patched_torch_load
            dataset = PygNodePropPredDataset(
                name=ogb_name, root=str(data_root), transform=transform
            )
        finally:
            torch.load = orig_torch_load

        data = dataset[0]
        split = dataset.get_idx_split()
        train_nodes = split["train"]
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_nodes] = True
        data.train_mask = train_mask
        return make_dataset_bundle(name, dataset, data, train_nodes)

    raise ValueError(f"unsupported dataset: {name}")


def make_loader(bundle: DatasetBundle, args: argparse.Namespace):
    from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader

    if args.loader == "neighbor":
        return NeighborLoader(
            bundle.data,
            input_nodes=bundle.train_nodes,
            num_neighbors=parse_fanout(args.fanout, args.num_layers),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    save_dir = Path(args.data_root) / "cluster_cache" / bundle.name
    cluster_data = ClusterData(
        bundle.data,
        num_parts=args.cluster_parts,
        recursive=False,
        save_dir=str(save_dir),
    )
    return ClusterLoader(
        cluster_data,
        batch_size=args.cluster_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )


def batch_train_mask(batch, device: torch.device) -> torch.Tensor:
    mask = getattr(batch, "train_mask", None)
    if mask is None:
        return torch.ones(batch.num_nodes, dtype=torch.bool, device=device)
    if mask.dim() > 1:
        mask = mask[:, 0]
    return mask


def train_neighbor_batch(
    model: GraphSAGE,
    optimizer: torch.optim.Optimizer,
    batch,
    device: torch.device,
) -> tuple[Optional[float], int]:
    batch = batch.to(device, non_blocking=True)
    seed_count = int(batch.batch_size)
    if seed_count == 0:
        return None, 0

    optimizer.zero_grad(set_to_none=True)
    out = model(batch.x, batch.edge_index)
    target = batch.y[:seed_count]
    loss = classification_loss(out[:seed_count], target)
    loss.backward()
    optimizer.step()
    return float(loss.detach()), seed_count


def train_cluster_batch(
    model: GraphSAGE,
    optimizer: torch.optim.Optimizer,
    batch,
    device: torch.device,
) -> tuple[Optional[float], int]:
    batch = batch.to(device, non_blocking=True)
    mask = batch_train_mask(batch, device)
    if int(mask.sum()) == 0:
        return None, 0

    optimizer.zero_grad(set_to_none=True)
    out = model(batch.x, batch.edge_index)
    target = batch.y[mask]
    loss = classification_loss(out[mask], target)
    loss.backward()
    optimizer.step()
    return float(loss.detach()), int(mask.sum())


def train_epoch(
    model: GraphSAGE,
    optimizer: torch.optim.Optimizer,
    loader,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> dict:
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

        if args.loader == "neighbor":
            loss_value, examples = train_neighbor_batch(model, optimizer, batch, device)
        else:
            loss_value, examples = train_cluster_batch(model, optimizer, batch, device)

        if loss_value is None:
            continue

        used_steps += 1
        total_examples += examples
        total_loss += loss_value * examples

        if args.log_every > 0 and used_steps % args.log_every == 0:
            avg_loss = total_loss / max(total_examples, 1)
            print(
                f"epoch={epoch:03d} step={used_steps:04d} "
                f"examples={total_examples} loss={avg_loss:.4f}"
            )

    if use_cuda_events:
        end_event.record()
        torch.cuda.synchronize()
        cuda_ms = start_event.elapsed_time(end_event)
    else:
        cuda_ms = 0.0

    wall_ms = (time.perf_counter() - start_wall) * 1000.0
    avg_loss = total_loss / max(total_examples, 1)
    return {
        "epoch": epoch,
        "steps": used_steps,
        "examples": total_examples,
        "loss": avg_loss,
        "wall_ms": wall_ms,
        "cuda_ms": cuda_ms,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    require_module("torch_geometric")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    bundle = load_dataset(
        args.dataset,
        Path(args.data_root),
        directed_ogbn_arxiv=args.directed_ogbn_arxiv,
    )
    loader = make_loader(bundle, args)

    model = GraphSAGE(
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
        f"dataset={args.dataset} loader={args.loader} "
        f"nodes={bundle.data.num_nodes} edges={bundle.data.num_edges} "
        f"features={bundle.data.num_features} classes={bundle.num_classes} "
        f"task={bundle.task_type}"
    )
    print(
        f"GraphSAGE layers={args.num_layers} hidden={args.hidden_channels} "
        f"dropout={args.dropout} batch={args.batch_size} fanout={args.fanout}"
    )

    rows = []
    for epoch in range(1, args.epochs + 1):
        row = train_epoch(model, optimizer, loader, args, device, epoch)
        row.update(
            {
                "dataset": args.dataset,
                "loader": args.loader,
                "num_layers": args.num_layers,
                "hidden_channels": args.hidden_channels,
                "batch_size": args.batch_size,
                "fanout": args.fanout,
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
    csv_path = output_dir / f"{args.dataset}_{args.loader}_graphsage_train.csv"
    write_rows(csv_path, rows)
    print(f"log_csv={csv_path}")

    if args.save_checkpoint:
        ckpt_path = output_dir / f"{args.dataset}_{args.loader}_graphsage.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"checkpoint={ckpt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
