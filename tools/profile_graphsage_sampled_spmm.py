#!/usr/bin/env python3
"""Profile SPMM share in full NeighborLoader + GraphSAGE training.

This standalone script profiles complete sampled GraphSAGE training epochs on
six datasets:

  ogbn_arxiv, ogbn_products, flickr, yelp, amazon_products, reddit

The reported percentages use CUDA compute time as the denominator; CPU loader
time and CUDA memcpy/memset events are excluded.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import ProfilerActivity, profile
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

DATASET_ALIASES = {
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

SPMM_LIKE_KEYWORDS = (
    "spmm",
    "sparse",
    "scatter_gather",
    "scatter",
    "segment",
    "gather_csr",
    "pyg::",
    "pyg_lib",
)

MESSAGE_PASSING_KEYWORDS = SPMM_LIKE_KEYWORDS + (
    "indexselect",
    "index_select",
    "index_select_large",
    "indexfunclargeindex",
)


@dataclass
class DatasetBundle:
    name: str
    data: object
    num_classes: int
    train_nodes: object
    task_type: str = "multiclass"


class GraphSAGE(nn.Module):
    """Fixed GraphSAGE model for NeighborLoader mini-batch training."""

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
    parser = argparse.ArgumentParser(
        description="Profile SPMM CUDA-time share for full GraphSAGE sampled training"
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--fanout", default="10,10")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", default="results/graphsage_sampled_profile")
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
    canonical = DATASET_ALIASES.get(key)
    if canonical is None:
        canonical = DATASET_ALIASES.get(key.replace("_", ""))
    if canonical is None:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise argparse.ArgumentTypeError(
            f"unsupported dataset: {name}; supported datasets: {supported}"
        )
    return canonical


def parse_datasets(value: str) -> list[str]:
    return [normalize_dataset_name(item) for item in value.split(",") if item.strip()]


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


def make_dataset_bundle(name: str, dataset, data, train_nodes) -> DatasetBundle:
    task_type = infer_task_type(data)
    return DatasetBundle(
        name=name,
        data=data,
        num_classes=infer_num_classes(dataset, data, task_type),
        train_nodes=train_nodes,
        task_type=task_type,
    )


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
    from torch_geometric.loader import NeighborLoader

    return NeighborLoader(
        bundle.data,
        input_nodes=bundle.train_nodes,
        num_neighbors=parse_fanout(args.fanout, args.num_layers),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )


def train_node_count(train_nodes) -> int:
    if torch.is_tensor(train_nodes):
        if train_nodes.dtype == torch.bool:
            return int(train_nodes.sum().item())
        return int(train_nodes.numel())
    return int(train_nodes.sum())


def classification_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if is_multilabel_target(target):
        return F.binary_cross_entropy_with_logits(target=target.to(logits.dtype), input=logits)
    return F.cross_entropy(logits, target.view(-1).long())


def train_step(model, optimizer, batch, device: torch.device) -> bool:
    batch = batch.to(device, non_blocking=True)
    seed_count = int(batch.batch_size)
    if seed_count == 0:
        return False

    optimizer.zero_grad(set_to_none=True)
    out = model(batch.x, batch.edge_index)
    loss = classification_loss(out[:seed_count], batch.y[:seed_count])
    loss.backward()
    optimizer.step()
    return True


def warmup(step_fn, loader: Iterable, warmup_steps: int, device: torch.device) -> None:
    iterator: Iterator | None = None
    completed = 0
    while completed < warmup_steps:
        if iterator is None:
            iterator = iter(loader)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if step_fn(batch):
            completed += 1
    if device.type == "cuda":
        torch.cuda.synchronize()


def event_us(event, attr: str) -> float:
    return float(getattr(event, attr, 0.0) or 0.0)


def cuda_time_attr(events: Sequence[object]) -> str:
    for attr in (
        "self_device_time_total",
        "self_cuda_time_total",
        "device_time_total",
        "cuda_time_total",
    ):
        if sum(event_us(event, attr) for event in events) > 0.0:
            return attr
    return "self_device_time_total"


def is_framework_wrapper(name: str) -> bool:
    lower = name.lower()
    return lower.startswith(
        (
            "aten::",
            "autograd::",
            "optimizer.",
            "profilerstep",
            "torch::",
            "torch_geometric::",
        )
    )


def is_cuda_compute_event(event) -> bool:
    if "cuda" not in str(getattr(event, "device_type", "")).lower():
        return False
    name = event.key.lower()
    if is_framework_wrapper(event.key):
        return False
    if "memcpy" in name or "memset" in name:
        return False
    return not any(
        token in name
        for token in (
            "activity buffer",
            "cudafree",
            "cudamalloc",
            "cudadevicegetattribute",
            "cudagetdriverentrypoint",
            "cudagetsymboladdress",
            "runtime triggered module loading",
            "lazy function loading",
        )
    )


def matches(name: str, keywords: Sequence[str]) -> bool:
    lower = name.lower()
    return any(keyword in lower for keyword in keywords)


def summarize_profile(events: Sequence[object], steps: int) -> dict:
    if steps <= 0:
        raise RuntimeError("no non-empty GraphSAGE training steps were profiled")

    attr = cuda_time_attr(events)
    cuda_events = [event for event in events if is_cuda_compute_event(event)]
    total_us = sum(event_us(event, attr) for event in cuda_events)

    def matched_us(keywords: Sequence[str]) -> float:
        return sum(
            event_us(event, attr)
            for event in cuda_events
            if matches(event.key, keywords)
        )

    spmm_us = matched_us(SPMM_LIKE_KEYWORDS)
    message_us = matched_us(MESSAGE_PASSING_KEYWORDS)
    return {
        "raw_compute_cuda_ms_per_step": total_us / 1000.0 / steps,
        "spmm_like_cuda_ms_per_step": spmm_us / 1000.0 / steps,
        "spmm_like_pct_raw_compute": 100.0 * spmm_us / total_us if total_us else 0.0,
        "message_passing_cuda_ms_per_step": message_us / 1000.0 / steps,
        "message_passing_pct_raw_compute": 100.0 * message_us / total_us
        if total_us
        else 0.0,
        "profiler_time_attr": attr,
    }


def run_dataset(dataset: str, args: argparse.Namespace, device: torch.device) -> dict:
    set_seed(args.seed)
    bundle = load_dataset(
        dataset,
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
    model.train()

    step_fn = lambda batch: train_step(model, optimizer, batch, device)
    warmup(step_fn, loader, args.warmup_steps, device)

    steps = 0
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.epochs):
            for batch in loader:
                if step_fn(batch):
                    steps += 1
                    prof.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    row = {
        "dataset": dataset,
        "loader": "neighbor",
        "model": "graphsage",
        "profile_mode": "full_training",
        "profile_epochs": args.epochs,
        "active_steps": steps,
        "profiled_steps": steps,
        "num_layers": args.num_layers,
        "hidden_channels": args.hidden_channels,
        "batch_size": args.batch_size,
        "fanout": args.fanout,
        "num_nodes": int(bundle.data.num_nodes),
        "num_edges": int(bundle.data.num_edges),
        "train_nodes": train_node_count(bundle.train_nodes),
        "warmup_steps": args.warmup_steps,
        **summarize_profile(prof.key_averages(), steps),
    }
    print(
        f"{dataset}: steps={steps} cuda={row['raw_compute_cuda_ms_per_step']:.3f} ms/step "
        f"spmm={row['spmm_like_pct_raw_compute']:.2f}% "
        f"message={row['message_passing_pct_raw_compute']:.2f}%"
    )
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    require_module("torch_geometric")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA-time profiling")

    rows = [run_dataset(dataset, args, device) for dataset in parse_datasets(args.datasets)]
    summary_csv = Path(args.output_dir) / "graphsage_neighbor_spmm_share_summary.csv"
    write_csv(summary_csv, rows)
    print(f"summary_csv={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
