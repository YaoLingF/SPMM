from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import load


CSRPack = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@lru_cache(maxsize=1)
def _load_ext():
    root = Path(__file__).resolve().parent
    return load(
        name="our_spmm_ext_src2",
        sources=[str(root / "spmm_ext.cpp"), str(root / "spmm_src2_cuda.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def preload_our_spmm_extension() -> None:
    """Compile/load the CUDA extension before timed training begins."""
    _load_ext()


def _assert_cuda_csr(rowptr: torch.Tensor, col: torch.Tensor, val: torch.Tensor) -> None:
    if not (rowptr.is_cuda and col.is_cuda and val.is_cuda):
        raise ValueError("CSR tensors must be CUDA tensors")
    if rowptr.dim() != 1 or col.dim() != 1 or val.dim() != 1:
        raise ValueError("rowptr, col, and val must be 1D tensors")
    if col.numel() != val.numel():
        raise ValueError("col and val length mismatch")
    if rowptr.dtype != col.dtype:
        raise ValueError("rowptr and col must have the same integer dtype")
    if rowptr.dtype not in (torch.int32, torch.int64):
        raise ValueError("rowptr/col must be int32 or int64")
    if val.dtype not in (torch.float16, torch.float32):
        raise ValueError("val must be float16 or float32")


def our_spmm(rowptr: torch.Tensor, col: torch.Tensor, val: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute CSR SpMM `out = A @ x`.

    `rowptr`, `col`, `val`, and `x` must already live on CUDA.  The CUDA
    extension returns FP32 output for both FP16 and FP32 input features.
    """
    _assert_cuda_csr(rowptr, col, val)
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dim() != 2:
        raise ValueError("x must be 2D [K, F]")
    # The src2 Tensor Core path is FP16 input / FP32 accumulate-output.  Casts
    # happen on GPU and keep the public PyTorch interface usable from FP32 PyG.
    if rowptr.dtype != torch.int32:
        rowptr = rowptr.to(torch.int32)
    if col.dtype != torch.int32:
        col = col.to(torch.int32)
    if val.dtype != torch.float16:
        val = val.to(torch.float16)
    if x.dtype != torch.float16:
        x = x.to(torch.float16)

    return _load_ext().forward(
        rowptr.contiguous(), col.contiguous(), val.contiguous(), x.contiguous()
    )


class OurSpMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rowptr: torch.Tensor,
        col: torch.Tensor,
        val: torch.Tensor,
        rowptr_t: torch.Tensor,
        col_t: torch.Tensor,
        val_t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(rowptr_t, col_t, val_t)
        return our_spmm(rowptr, col, val, x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        rowptr_t, col_t, val_t = ctx.saved_tensors
        grad_x = our_spmm(rowptr_t, col_t, val_t, grad_out.contiguous())
        return None, None, None, None, None, None, grad_x


def _index2ptr(index: torch.Tensor, size: int) -> torch.Tensor:
    counts = torch.bincount(index, minlength=size)
    rowptr = torch.empty(size + 1, device=index.device, dtype=torch.int64)
    rowptr[0] = 0
    rowptr[1:] = torch.cumsum(counts, dim=0)
    return rowptr


def _csr_from_edges(
    row: torch.Tensor,
    col: torch.Tensor,
    weight: torch.Tensor,
    num_rows: int,
    num_cols_for_sort: int,
    index_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Sort by row first, then col for deterministic CSR.
    perm = torch.argsort(row.to(torch.int64) * int(num_cols_for_sort) + col.to(torch.int64))
    row = row[perm]
    col = col[perm]
    weight = weight[perm]
    rowptr = _index2ptr(row.to(torch.int64), num_rows)
    return (
        rowptr.to(index_dtype).contiguous(),
        col.to(index_dtype).contiguous(),
        weight.contiguous(),
    )


@torch.no_grad()
def edge_index_weight_to_csr_pack(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
    value_dtype: torch.dtype = torch.float16,
    index_dtype: torch.dtype = torch.int32,
) -> CSRPack:
    """Build forward CSR and transpose CSR for weighted PyG `edge_index`.

    `edge_index[0] = src`, `edge_index[1] = dst`, and `edge_weight[e]` is the
    value used for `out[dst] += edge_weight[e] * x[src]`.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_weight.dim() != 1 or edge_weight.numel() != edge_index.size(1):
        raise ValueError("edge_weight must be 1D with length E")
    if not edge_index.is_cuda or not edge_weight.is_cuda:
        raise ValueError("edge_index and edge_weight must be on CUDA")
    if index_dtype not in (torch.int32, torch.int64):
        raise ValueError("index_dtype must be torch.int32 or torch.int64")

    src = edge_index[0].to(torch.int64)
    dst = edge_index[1].to(torch.int64)
    weight = edge_weight.to(value_dtype)
    rowptr, col, val = _csr_from_edges(
        row=dst,
        col=src,
        weight=weight,
        num_rows=num_nodes,
        num_cols_for_sort=num_nodes,
        index_dtype=index_dtype,
    )
    rowptr_t, col_t, val_t = _csr_from_edges(
        row=src,
        col=dst,
        weight=weight,
        num_rows=num_nodes,
        num_cols_for_sort=num_nodes,
        index_dtype=index_dtype,
    )
    return rowptr, col, val, rowptr_t, col_t, val_t


@torch.no_grad()
def edge_index_to_csr_pack(
    edge_index: torch.Tensor,
    num_nodes: int,
    mean: bool = True,
    value_dtype: torch.dtype = torch.float32,
    index_dtype: torch.dtype = torch.int32,
) -> CSRPack:
    """Build forward CSR and transpose CSR for PyG `edge_index`.

    PyG convention is `edge_index[0] = src`, `edge_index[1] = dst`.  The forward
    CSR represents `out[dst] += value * x[src]`.

    The transpose CSR reuses exactly the same edge weights, only reordered for
    `grad_x = A.T @ grad_out`.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if not edge_index.is_cuda:
        raise ValueError("edge_index must be on CUDA before CSR construction")
    if index_dtype not in (torch.int32, torch.int64):
        raise ValueError("index_dtype must be torch.int32 or torch.int64")

    src = edge_index[0].to(torch.int64)
    dst = edge_index[1].to(torch.int64)

    if mean:
        deg = torch.bincount(dst, minlength=num_nodes).clamp_min_(1)
        weight = (1.0 / deg[dst].to(torch.float32)).to(value_dtype)
    else:
        weight = torch.ones(src.numel(), device=edge_index.device, dtype=value_dtype)

    rowptr, col, val = _csr_from_edges(
        row=dst,
        col=src,
        weight=weight,
        num_rows=num_nodes,
        num_cols_for_sort=num_nodes,
        index_dtype=index_dtype,
    )
    rowptr_t, col_t, val_t = _csr_from_edges(
        row=src,
        col=dst,
        weight=weight,
        num_rows=num_nodes,
        num_cols_for_sort=num_nodes,
        index_dtype=index_dtype,
    )
    return rowptr, col, val, rowptr_t, col_t, val_t
