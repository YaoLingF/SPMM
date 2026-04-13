#!/usr/bin/env python3
"""Export sampled GNN batches and save workload artifacts for stage-1."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


def parse_int_list(raw: str) -> List[int]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise ValueError("fanouts list is empty")
    return [int(x) for x in parts]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_to_int_list(x: object) -> List[int]:
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return []


def get_dataset(name: str, root: str):
    ds = name.lower()

    if ds == "synthetic":
        from torch_geometric.data import Data

        num_nodes = 20000
        feat_dim = 128
        mean_degree = 12
        num_edges = num_nodes * mean_degree

        src = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
        dst = torch.randint(0, num_nodes, (num_edges,), dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        x = torch.randn(num_nodes, feat_dim, dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index)

        perm = torch.randperm(num_nodes)
        n_train = int(0.6 * num_nodes)
        n_val = int(0.2 * num_nodes)
        split_idx = {
            "train": perm[:n_train],
            "val": perm[n_train : n_train + n_val],
            "test": perm[n_train + n_val :],
        }
        return data, split_idx, "synthetic"

    if ds == "reddit":
        from torch_geometric.datasets import Reddit

        data = Reddit(root=root)[0]
        split_idx = {
            "train": torch.where(data.train_mask)[0],
            "val": torch.where(data.val_mask)[0],
            "test": torch.where(data.test_mask)[0],
        }
        return data, split_idx, "reddit"

    if ds == "yelp":
        from torch_geometric.datasets import Yelp

        data = Yelp(root=root)[0]
        split_idx = {
            "train": torch.where(data.train_mask)[0],
            "val": torch.where(data.val_mask)[0],
            "test": torch.where(data.test_mask)[0],
        }
        return data, split_idx, "yelp"

    if ds in {"ogbn-arxiv", "ogbn-products"}:
        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name=ds, root=root)
        data = dataset[0]
        idx = dataset.get_idx_split()
        split_idx = {"train": idx["train"], "val": idx["valid"], "test": idx["test"]}
        return data, split_idx, ds

    raise ValueError(
        "Unsupported dataset. Use: synthetic, reddit, yelp, ogbn-arxiv, ogbn-products"
    )


def build_loader(data, input_nodes, fanouts, batch_size, shuffle, num_workers):
    from torch_geometric.loader import NeighborLoader

    kwargs = {
        "data": data,
        "input_nodes": input_nodes,
        "num_neighbors": fanouts,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    try:
        return NeighborLoader(**kwargs)
    except TypeError:
        kwargs.pop("persistent_workers", None)
        return NeighborLoader(**kwargs)


def iter_synthetic_batches(data, input_nodes, fanouts, batch_size, num_batches):
    """Fallback synthetic batch iterator without neighbor sampling backend."""
    from torch_geometric.data import Data

    edge_index = data.edge_index.cpu()
    src = edge_index[0]
    dst = edge_index[1]
    num_nodes_total = int(data.num_nodes)

    input_nodes = input_nodes.cpu()
    order = input_nodes[torch.randperm(input_nodes.numel())]
    cursor = 0
    expand_factor = max(2, int(np.sqrt(max(1, sum(fanouts)))))

    for _ in range(num_batches):
        if cursor + batch_size > order.numel():
            order = input_nodes[torch.randperm(input_nodes.numel())]
            cursor = 0
        seeds = order[cursor : cursor + batch_size]
        cursor += batch_size

        n_extra = min(num_nodes_total, max(batch_size, int(seeds.numel() * expand_factor)))
        extra_nodes = torch.randint(0, num_nodes_total, (n_extra,), dtype=torch.long)
        selected = torch.unique(torch.cat([seeds, extra_nodes], dim=0))

        selected_mask = torch.zeros(num_nodes_total, dtype=torch.bool)
        selected_mask[selected] = True
        mask = selected_mask[src] & selected_mask[dst]
        sub_src = src[mask]
        sub_dst = dst[mask]

        id_map = torch.full((num_nodes_total,), -1, dtype=torch.long)
        id_map[selected] = torch.arange(selected.numel(), dtype=torch.long)
        sub_edge_index = torch.stack([id_map[sub_src], id_map[sub_dst]], dim=0)
        x_sub = data.x[selected] if getattr(data, "x", None) is not None else None

        batch = Data(x=x_sub, edge_index=sub_edge_index)
        batch.batch_size = int(seeds.numel())
        yield batch


def compute_csr(edge_index_np: np.ndarray, num_rows: int, row_mode: str):
    src = edge_index_np[0].astype(np.int64, copy=False)
    dst = edge_index_np[1].astype(np.int64, copy=False)

    if row_mode == "dst":
        rows = dst
        cols = src
    else:
        rows = src
        cols = dst

    valid = (rows >= 0) & (rows < num_rows) & (cols >= 0)
    rows = rows[valid]
    cols = cols[valid]

    order = np.argsort(rows, kind="stable")
    rows_sorted = rows[order]
    cols_sorted = cols[order]

    row_nnz = np.bincount(rows_sorted, minlength=num_rows).astype(np.int64, copy=False)
    row_ptr = np.empty(num_rows + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(row_nnz, out=row_ptr[1:])
    return row_ptr, cols_sorted, row_nnz


def summarize_row_nnz(row_nnz: np.ndarray, long_threshold: int, tc_chunk: int) -> Dict[str, float]:
    num_rows = int(row_nnz.size)
    nnz = int(row_nnz.sum())

    avg = float(nnz / num_rows) if num_rows > 0 else 0.0
    p50 = float(np.percentile(row_nnz, 50)) if num_rows > 0 else 0.0
    p95 = float(np.percentile(row_nnz, 95)) if num_rows > 0 else 0.0
    p99 = float(np.percentile(row_nnz, 99)) if num_rows > 0 else 0.0
    max_row_nnz = int(np.max(row_nnz)) if num_rows > 0 else 0

    active = row_nnz[row_nnz > 0]
    active_avg = float(active.mean()) if active.size > 0 else 0.0
    active_p95 = float(np.percentile(active, 95)) if active.size > 0 else 0.0
    active_p99 = float(np.percentile(active, 99)) if active.size > 0 else 0.0
    active_max = int(np.max(active)) if active.size > 0 else 0

    long_ratio = float(np.mean(row_nnz >= long_threshold)) if num_rows > 0 else 0.0
    active_row_ratio = float(np.mean(row_nnz > 0)) if num_rows > 0 else 0.0

    if nnz > 0:
        chunkable_nnz = int(np.sum((row_nnz // tc_chunk) * tc_chunk))
        chunkable_ratio = float(chunkable_nnz / nnz)
    else:
        chunkable_ratio = 0.0
    residue_ratio = 1.0 - chunkable_ratio

    return {
        "num_rows": num_rows,
        "nnz": nnz,
        "avg_row_nnz": avg,
        "p50_row_nnz": p50,
        "p95_row_nnz": p95,
        "p99_row_nnz": p99,
        "max_row_nnz": max_row_nnz,
        "active_avg_row_nnz": active_avg,
        "active_p95_row_nnz": active_p95,
        "active_p99_row_nnz": active_p99,
        "active_max_row_nnz": active_max,
        "long_row_ratio": long_ratio,
        "chunkable_ratio": chunkable_ratio,
        "residue_ratio": residue_ratio,
        "active_row_ratio": active_row_ratio,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export sampled mini-batches and workload metrics")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="graphsage", choices=["gcn", "graphsage"])
    parser.add_argument("--fanouts", required=True, help="Comma-separated fanouts, e.g. 25,10")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default="./datasets")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--row-mode", default="dst", choices=["dst", "src"])
    parser.add_argument("--long-threshold", type=int, default=64)
    parser.add_argument("--tc-chunk", type=int, default=8)
    parser.add_argument("--save-row-nnz", action="store_true", default=True)
    parser.add_argument("--no-save-row-nnz", dest="save_row_nnz", action="store_false")
    parser.add_argument("--save-csr", action="store_true", default=False)
    parser.add_argument("--save-edge-index", action="store_true", default=False)
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.set_defaults(shuffle=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fanouts = parse_int_list(args.fanouts)
    set_seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = out_dir / "rows"
    csr_dir = out_dir / "csr"
    edge_dir = out_dir / "edge_index"
    if args.save_row_nnz:
        rows_dir.mkdir(parents=True, exist_ok=True)
    if args.save_csr:
        csr_dir.mkdir(parents=True, exist_ok=True)
    if args.save_edge_index:
        edge_dir.mkdir(parents=True, exist_ok=True)

    data, split_idx_map, dataset_tag = get_dataset(args.dataset, args.root)
    input_nodes = split_idx_map[args.split]

    if dataset_tag == "synthetic":
        batch_iter: Iterable[torch.Tensor] = iter_synthetic_batches(
            data=data,
            input_nodes=input_nodes,
            fanouts=fanouts,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
        )
    else:
        loader = build_loader(
            data=data,
            input_nodes=input_nodes,
            fanouts=fanouts,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
        )
        batch_iter = loader

    fanout_tag = ",".join(str(f) for f in fanouts)
    records = []
    layer_records = []
    start = time.time()

    try:
        for batch_id, batch in enumerate(batch_iter):
            if batch_id >= args.num_batches:
                break

            edge_index = batch.edge_index.detach().cpu().numpy()
            num_nodes = int(batch.num_nodes)
            row_ptr, col_idx, row_nnz = compute_csr(
                edge_index_np=edge_index, num_rows=num_nodes, row_mode=args.row_mode
            )
            stats = summarize_row_nnz(
                row_nnz=row_nnz, long_threshold=args.long_threshold, tc_chunk=args.tc_chunk
            )

            row_nnz_rel = ""
            if args.save_row_nnz:
                row_file = rows_dir / f"batch_{batch_id:06d}.npy"
                np.save(row_file, row_nnz)
                row_nnz_rel = str(row_file.relative_to(out_dir))

            csr_rel = ""
            if args.save_csr:
                csr_file = csr_dir / f"batch_{batch_id:06d}.npz"
                np.savez_compressed(csr_file, row_ptr=row_ptr, col_idx=col_idx)
                csr_rel = str(csr_file.relative_to(out_dir))

            dense_width = -1
            if getattr(batch, "x", None) is not None and batch.x.dim() == 2:
                dense_width = int(batch.x.size(1))

            edge_rel = ""
            if args.save_edge_index:
                edge_file = edge_dir / f"batch_{batch_id:06d}.npz"
                sampled_nodes_by_hop = tensor_to_int_list(getattr(batch, "num_sampled_nodes", None))
                sampled_edges_by_hop = tensor_to_int_list(getattr(batch, "num_sampled_edges", None))
                payload = {
                    "edge_index": edge_index.astype(np.int64, copy=False),
                    "batch_id": np.array([batch_id], dtype=np.int64),
                    "num_nodes": np.array([num_nodes], dtype=np.int64),
                    "num_edges": np.array([int(batch.num_edges)], dtype=np.int64),
                    "feature_width": np.array([dense_width], dtype=np.int64),
                    "seed_nodes": np.array([int(getattr(batch, "batch_size", 0))], dtype=np.int64),
                }
                if sampled_nodes_by_hop:
                    payload["num_sampled_nodes_by_hop"] = np.array(sampled_nodes_by_hop, dtype=np.int64)
                if sampled_edges_by_hop:
                    payload["num_sampled_edges_by_hop"] = np.array(sampled_edges_by_hop, dtype=np.int64)
                if hasattr(batch, "n_id"):
                    payload["n_id"] = batch.n_id.detach().cpu().numpy().astype(np.int64, copy=False)
                if hasattr(batch, "e_id"):
                    payload["e_id"] = batch.e_id.detach().cpu().numpy().astype(np.int64, copy=False)
                np.savez_compressed(edge_file, **payload)
                edge_rel = str(edge_file.relative_to(out_dir))

                num_layer_rows = max(len(fanouts), len(sampled_edges_by_hop))
                for layer_id in range(num_layer_rows):
                    dst_nodes = sampled_nodes_by_hop[layer_id] if layer_id < len(sampled_nodes_by_hop) else -1
                    src_nodes = (
                        sampled_nodes_by_hop[layer_id + 1]
                        if (layer_id + 1) < len(sampled_nodes_by_hop)
                        else -1
                    )
                    layer_records.append(
                        {
                            "dataset": dataset_tag,
                            "model": args.model,
                            "fanouts": fanout_tag,
                            "split": args.split,
                            "batch_id": batch_id,
                            "layer_id": layer_id,
                            "sampled_nodes_layer": src_nodes if src_nodes >= 0 else dst_nodes,
                            "sampled_nodes_src_layer": src_nodes,
                            "sampled_nodes_dst_layer": dst_nodes,
                            "sampled_edges_layer": (
                                sampled_edges_by_hop[layer_id]
                                if layer_id < len(sampled_edges_by_hop)
                                else -1
                            ),
                            "edge_index_file": edge_rel,
                        }
                    )

            rec = {
                "dataset": dataset_tag,
                "model": args.model,
                "fanouts": fanout_tag,
                "num_layers": len(fanouts),
                "split": args.split,
                "batch_id": batch_id,
                "seed_nodes": int(getattr(batch, "batch_size", 0)),
                "sampled_nodes": int(batch.num_nodes),
                "sampled_edges": int(batch.num_edges),
                "long_row_threshold": args.long_threshold,
                "tc_chunk": args.tc_chunk,
                "dense_width": dense_width,
                "row_mode": args.row_mode,
                "row_nnz_file": row_nnz_rel,
                "csr_file": csr_rel,
                "edge_index_file": edge_rel,
            }
            rec.update(stats)
            records.append(rec)

            if (batch_id + 1) % 25 == 0:
                print(f"processed {batch_id + 1} batches")
    except ImportError as exc:
        msg = str(exc)
        if "pyg-lib" in msg or "torch-sparse" in msg:
            raise RuntimeError(
                "Neighbor sampling backend missing. Install pyg-lib/torch-sparse, or use --dataset synthetic."
            ) from exc
        raise

    metrics_df = pd.DataFrame(records)
    metrics_csv = out_dir / "batch_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    layer_meta_csv = ""
    if layer_records:
        layer_meta_path = out_dir / "batch_layer_meta.csv"
        pd.DataFrame(layer_records).to_csv(layer_meta_path, index=False)
        layer_meta_csv = str(layer_meta_path.name)

    meta = {
        "dataset": dataset_tag,
        "model": args.model,
        "fanouts": fanouts,
        "split": args.split,
        "batch_size": args.batch_size,
        "num_batches_requested": args.num_batches,
        "num_batches_exported": int(len(records)),
        "row_mode": args.row_mode,
        "long_row_threshold": args.long_threshold,
        "tc_chunk": args.tc_chunk,
        "save_row_nnz": args.save_row_nnz,
        "save_csr": args.save_csr,
        "save_edge_index": args.save_edge_index,
        "seed": args.seed,
        "elapsed_sec": time.time() - start,
        "metrics_csv": str(metrics_csv.name),
        "layer_meta_csv": layer_meta_csv,
    }
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"export finished: {len(records)} batches")
    print(f"metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
