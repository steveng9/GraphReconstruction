"""GNN classifier attack: a GNN trained on the synthetic graph to predict each hidden
attribute from the known ones, applied inductively to the target graph."""
from __future__ import annotations

import numpy as np
import torch

from graphrecon.attacks import register_attack
from graphrecon.attacks.base import normalize_rows
from graphrecon.attacks.synth.node.classifiers import run_node_model
from graphrecon.data.features import node_features
from graphrecon.data.graph import MISSING
from graphrecon.releases.gnn.models import MODELS

GNN_CLASSIFIER_DEFAULTS = {
    "arch": "SAGE", "arch_params": {},        # any registered GNN (releases/gnn/models.py)
    "features": ["qi", "structure"],          # neighbourhood aggregation is the GNN's job
    "lr": 0.01, "weight_decay": 5e-4, "epochs": 300, "patience": 40, "val_frac": 0.2,
}


class GNNNodeModel:
    def __init__(self, params: dict, seed: int):
        self.p, self.seed = params, seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _x(self, graph):
        X = node_features(graph, self.qi, self.p["features"])
        return torch.as_tensor((X - self.mu) / self.sd, dtype=torch.float32, device=self.device)

    def fit(self, graph, qi, labels, card, mask=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.qi, self.card = qi, card
        X = node_features(graph, qi, self.p["features"])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        x = self._x(graph)
        ei = torch.as_tensor(graph.directed_edges(), dtype=torch.long, device=self.device)
        ok = labels != MISSING if mask is None else (mask & (labels != MISSING))
        idx = rng.permutation(np.flatnonzero(ok))
        n_val = int(self.p["val_frac"] * len(idx))
        val, tr = torch.as_tensor(idx[:n_val], device=self.device), torch.as_tensor(idx[n_val:], device=self.device)
        y = torch.as_tensor(np.maximum(labels, 0), device=self.device)
        arch_p = MODELS.resolve(self.p["arch"], self.p["arch_params"])
        self.model = MODELS.get(self.p["arch"]).fn(x.shape[1], card, **arch_p).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.p["lr"], weight_decay=self.p["weight_decay"])
        best, best_state, since = float("inf"), None, 0
        for _ in range(self.p["epochs"]):
            self.model.train(); opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.model(x, ei)[tr], y[tr])
            loss.backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = float(torch.nn.functional.cross_entropy(self.model(x, ei)[val], y[val])) if n_val else float(loss)
            if vl < best:
                best, since = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                since += 1
                if since >= self.p["patience"]:
                    break
        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, graph):
        ei = torch.as_tensor(graph.directed_edges(), dtype=torch.long, device=self.device)
        P = torch.softmax(self.model(self._x(graph), ei), -1).cpu().numpy()
        return normalize_rows(P, 1e-4)


@register_attack("synth", "node", "GNNClassifier", defaults=GNN_CLASSIFIER_DEFAULTS)
def gnn_classifier(release, view, rng, **params):
    """GNN trained on the synthetic graph (hidden | known + structure), applied inductively."""
    seed = int(rng.integers(2**31))
    return run_node_model(lambda: GNNNodeModel(params, seed), release, view)
