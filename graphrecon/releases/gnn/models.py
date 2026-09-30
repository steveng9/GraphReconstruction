"""GNN architectures (PyG). All share the signature ``(in_dim, out_dim, **params)``."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from graphrecon.registry import Registry

MODELS = Registry("gnn model")


class GNN(nn.Module):
    """Stack of message-passing layers with ReLU + dropout between them."""

    def __init__(self, convs: list[nn.Module], dropout: float):
        super().__init__()
        self.convs = nn.ModuleList(convs)
        self.dropout = dropout

    def forward(self, x, edge_index, return_hidden: bool = False):
        hidden = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
                hidden.append(x)
                x = nn.functional.dropout(x, self.dropout, self.training)
        return (x, hidden) if return_hidden else x


def _dims(in_dim, out_dim, hidden, layers):
    return [in_dim] + [hidden] * (layers - 1) + [out_dim]


@MODELS.register("GCN", defaults={"hidden": 64, "layers": 2, "dropout": 0.5})
def gcn(in_dim, out_dim, hidden, layers, dropout):
    """Graph convolutional network (Kipf & Welling 2017)."""
    d = _dims(in_dim, out_dim, hidden, layers)
    return GNN([GCNConv(d[i], d[i + 1]) for i in range(layers)], dropout)


@MODELS.register("SAGE", defaults={"hidden": 64, "layers": 2, "dropout": 0.5, "aggr": "mean"})
def sage(in_dim, out_dim, hidden, layers, dropout, aggr):
    """GraphSAGE (Hamilton et al. 2017), inductive."""
    d = _dims(in_dim, out_dim, hidden, layers)
    return GNN([SAGEConv(d[i], d[i + 1], aggr=aggr) for i in range(layers)], dropout)


@MODELS.register("GAT", defaults={"hidden": 16, "heads": 4, "layers": 2, "dropout": 0.5})
def gat(in_dim, out_dim, hidden, heads, layers, dropout):
    """Graph attention network (Velickovic et al. 2018); hidden width is per head."""
    convs, d_in = [], in_dim
    for i in range(layers):
        last = i == layers - 1
        convs.append(GATConv(d_in, out_dim if last else hidden, heads=1 if last else heads,
                             dropout=dropout))
        d_in = hidden * heads
    return GNN(convs, dropout)
