"""Train a node classifier on the private graph (full batch, early stopping on validation)."""
from __future__ import annotations

import copy

import numpy as np
import torch

from graphrecon.data.graph import MISSING, AttributedGraph
from graphrecon.releases.gnn.encoding import InputEncoder
from graphrecon.releases.gnn.models import MODELS
from graphrecon.releases.gnn.release import GNNRelease

TRAIN_DEFAULTS = {
    "label": None,            # attribute the model predicts (required)
    "inputs": None,           # input attributes; None = every other attribute
    "use_x": True,            # also feed dense features x if the graph has them
    "val_frac": 0.2,          # of labelled train-graph nodes, for early stopping
    "lr": 0.01, "weight_decay": 5e-4, "epochs": 300, "patience": 50,
}


def _accuracy(logits, y, mask):
    return float((logits[mask].argmax(-1) == y[mask]).float().mean()) if mask.any() else float("nan")


def train_gnn(graph: AttributedGraph, arch: str, arch_params: dict, train: dict, seed: int,
              eval_graphs: dict[str, AttributedGraph] | None = None, device: str | None = None):
    """Returns (GNNRelease, metrics). ``eval_graphs`` are scored inductively (utility)."""
    t = {**TRAIN_DEFAULTS, **train}
    if t["label"] is None:
        raise ValueError("GNN training needs a 'label' attribute")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    inputs = t["inputs"] or [a for a in graph.attr_names if a != t["label"]]
    use_x = bool(t["use_x"] and graph.x is not None)
    enc = InputEncoder(graph.schema, inputs, use_x, graph.x.shape[1] if use_x else 0)
    model = MODELS.get(arch).fn(enc.dim, graph.card(t["label"]), **arch_params).to(device)

    x = torch.as_tensor(enc.encode(graph), device=device)
    ei = torch.as_tensor(graph.directed_edges(), dtype=torch.long, device=device)
    y_np = graph.col(t["label"])
    y = torch.as_tensor(np.maximum(y_np, 0), device=device)
    labelled = np.flatnonzero(y_np != MISSING)
    perm = rng.permutation(labelled)
    n_val = int(round(t["val_frac"] * len(perm)))
    val_mask = torch.zeros(graph.num_nodes, dtype=torch.bool, device=device)
    tr_mask = torch.zeros_like(val_mask)
    val_mask[perm[:n_val]] = True
    tr_mask[perm[n_val:]] = True

    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    best, best_state, since = -1.0, None, 0
    for epoch in range(t["epochs"]):
        model.train()
        opt.zero_grad()
        out = model(x, ei)
        loss = torch.nn.functional.cross_entropy(out[tr_mask], y[tr_mask])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = _accuracy(model(x, ei), y, val_mask)
        if val > best:
            best, best_state, since = val, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= t["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(x, ei)
    metrics = {"train_acc": _accuracy(out, y, tr_mask), "val_acc": _accuracy(out, y, val_mask),
               "epochs": epoch + 1}
    release = GNNRelease(model, enc, t["label"], graph.schema[t["label"]], arch, arch_params,
                         server_graph=graph, device=device)
    for name, g in (eval_graphs or {}).items():
        yy = g.col(t["label"])
        m = yy != MISSING
        metrics[f"{name}_acc"] = float((release.query_graph(g).argmax(1)[m] == yy[m]).mean())
    return release, metrics
