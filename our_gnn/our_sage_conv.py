from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .our_spmm import CSRPack, OurSpMM, edge_index_to_csr_pack


class OurSAGEConv(nn.Module):
    """GraphSAGE mean aggregation using the custom CSR SpMM extension."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        csr_pack: Optional[CSRPack] = None,
    ) -> torch.Tensor:
        if csr_pack is None:
            if edge_index is None:
                raise ValueError("either edge_index or csr_pack must be provided")
            csr_pack = edge_index_to_csr_pack(
                edge_index=edge_index,
                num_nodes=x.size(0),
                mean=True,
                value_dtype=torch.float16,
                index_dtype=torch.int32,
            )

        rowptr, col, val, rowptr_t, col_t, val_t = csr_pack
        agg = OurSpMM.apply(rowptr, col, val, rowptr_t, col_t, val_t, x)
        return self.lin_l(agg) + self.lin_r(x)


def build_csr_pack(edge_index: torch.Tensor, num_nodes: int) -> CSRPack:
    return edge_index_to_csr_pack(
        edge_index=edge_index,
        num_nodes=num_nodes,
        mean=True,
        value_dtype=torch.float16,
        index_dtype=torch.int32,
    )
