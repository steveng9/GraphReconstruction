"""The published GNN and the access modes an adversary may be granted.

* ``query_nodes(ids)``  -- black-box, *transductive*: the server evaluates the model
  on its private graph and returns posteriors for the requested nodes (the
  GNN-as-a-service setting of link stealing, He et al. 2021).
* ``query_graph(g)``    -- black-box, *inductive*: the adversary supplies a graph
  with the model's input attributes and receives posteriors (model inversion).
* ``model`` / ``encoder`` -- white-box access.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from graphrecon.data.graph import AttributedGraph
from graphrecon.releases.gnn.encoding import InputEncoder
from graphrecon.releases.gnn.models import MODELS


class GNNRelease:
    def __init__(self, model, encoder: InputEncoder, label_attr: str, label_schema: list[str],
                 arch: str, arch_params: dict, server_graph: AttributedGraph | None = None,
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.encoder, self.label_attr, self.label_schema = encoder, label_attr, label_schema
        self.arch, self.arch_params = arch, arch_params
        self._server_graph = server_graph
        self._server_probs = None

    @property
    def input_attrs(self) -> list[str]:
        return self.encoder.input_attrs

    @property
    def num_classes(self) -> int:
        return len(self.label_schema)

    @torch.no_grad()
    def _forward(self, X: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(X, device=self.device)
        ei = torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        return torch.softmax(self.model(x, ei), dim=-1).cpu().numpy()

    def query_graph(self, graph: AttributedGraph, soft: dict | None = None) -> np.ndarray:
        missing = set(self.input_attrs) - set(graph.attr_names) - set(soft or {})
        if missing:
            raise ValueError(f"query graph lacks model inputs {sorted(missing)}")
        return self._forward(self.encoder.encode(graph, soft), graph.directed_edges())

    def query_nodes(self, nodes: np.ndarray | None = None) -> np.ndarray:
        if self._server_graph is None:
            raise RuntimeError("no server graph attached; transductive queries unavailable")
        if self._server_probs is None:
            self._server_probs = self.query_graph(self._server_graph)
        return self._server_probs if nodes is None else self._server_probs[np.asarray(nodes)]

    def attach_server_graph(self, graph: AttributedGraph):
        self._server_graph, self._server_probs = graph, None

    # -- persistence --------------------------------------------------------
    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), d / "model.pt")
        with open(d / "spec.json", "w") as f:
            json.dump({"encoder": self.encoder.to_dict(), "label_attr": self.label_attr,
                       "label_schema": self.label_schema, "arch": self.arch,
                       "arch_params": self.arch_params}, f, indent=1)

    @classmethod
    def load(cls, directory, server_graph=None, device=None) -> "GNNRelease":
        d = Path(directory)
        spec = json.loads((d / "spec.json").read_text())
        enc = InputEncoder.from_dict(spec["encoder"])
        model = MODELS.get(spec["arch"]).fn(enc.dim, len(spec["label_schema"]), **spec["arch_params"])
        model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True))
        return cls(model, enc, spec["label_attr"], spec["label_schema"], spec["arch"],
                   spec["arch_params"], server_graph, device)

    @staticmethod
    def exists(directory) -> bool:
        return (Path(directory) / "model.pt").exists()
