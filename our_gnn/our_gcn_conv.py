from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from .our_spmm import CSRPack, OurSpMM, edge_index_weight_to_csr_pack


class OurGCNConv(nn.Module):
    """GCNConv-compatible layer backed by the custom CSR SpMM extension."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        csr_pack: Optional[CSRPack] = None,
    ) -> torch.Tensor:
        if csr_pack is None:
            if edge_index is None:
                raise ValueError("either edge_index or csr_pack must be provided")
            norm_edge_index, norm_edge_weight = gcn_norm(
                edge_index,
                edge_weight=None,
                num_nodes=x.size(0),
                improved=False,
                add_self_loops=True,
                flow="source_to_target",
                dtype=x.dtype,
            )
            csr_pack = edge_index_weight_to_csr_pack(
                norm_edge_index,
                norm_edge_weight,
                num_nodes=x.size(0),
                value_dtype=torch.float16,
                index_dtype=torch.int32,
            )

        x = self.lin(x)
        rowptr, col, val, rowptr_t, col_t, val_t = csr_pack
        out = OurSpMM.apply(rowptr, col, val, rowptr_t, col_t, val_t, x)
        if self.bias is not None:
            out = out + self.bias
        return out


@torch.no_grad()
def build_gcn_csr_pack(edge_index: torch.Tensor, num_nodes: int, dtype: torch.dtype = torch.float32) -> CSRPack:
    norm_edge_index, norm_edge_weight = gcn_norm(
        edge_index,
        edge_weight=None,
        num_nodes=num_nodes,
        improved=False,
        add_self_loops=True,
        flow="source_to_target",
        dtype=dtype,
    )
    return edge_index_weight_to_csr_pack(
        norm_edge_index,
        norm_edge_weight,
        num_nodes=num_nodes,
        value_dtype=torch.float16,
        index_dtype=torch.int32,
    )
