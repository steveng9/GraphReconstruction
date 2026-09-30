"""Content-addressed storage of releases.

A release is identified by everything that determines it:

    <kind>/<dataset key>/<split tag>/s<sample>/<method>-<hash(params, seed)>

and stored at workspace/releases/<release id>/ with a ``meta.json`` recording the
full configuration, provenance and a quality / utility report. ``get_or_create_*``
return the cached release when it exists, so experiments are idempotent.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np

from graphrecon.config import git_commit, param_hash
from graphrecon.data.graph import AttributedGraph
from graphrecon.data.splits import Split
from graphrecon.paths import RELEASES


def split_tag(split_cfg: dict) -> str:
    return f"n{split_cfg['n_parts']}-seed{split_cfg['seed']}"


def release_id(kind, dataset_key, split_cfg, sample, method, params, seed) -> str:
    h = param_hash({"params": params, "seed": seed}, 8)
    return f"{kind}/{dataset_key}/{split_tag(split_cfg)}/s{sample}/{method}-{h}"


def _write_meta(path, meta):
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)


def read_meta(rid: str) -> dict:
    return json.loads((RELEASES / rid / "meta.json").read_text())


def _base_meta(rid, kind, method, params, seed, dataset_key, split_cfg, sample):
    return {"release_id": rid, "kind": kind, "method": method, "params": params, "seed": seed,
            "dataset_key": dataset_key, "split": split_cfg, "sample": sample,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit()}


def get_or_create_synth(dataset_key: str, split_cfg: dict, split: Split, method: str,
                        params: dict | None, seed: int, force: bool = False):
    """Returns (release_id, synthetic AttributedGraph, meta)."""
    from graphrecon.releases.synth import GENERATORS
    from graphrecon.releases.synth.quality import quality_report

    p = GENERATORS.resolve(method, params)
    rid = release_id("synth", dataset_key, split_cfg, split.sample, method, p, seed)
    path = RELEASES / rid
    if not force and AttributedGraph.exists(path) and (path / "meta.json").exists():
        return rid, AttributedGraph.load(path), read_meta(rid)
    t0 = time.time()
    rng = np.random.default_rng([seed, split.sample])
    syn = GENERATORS.get(method).fn(split["train"], rng, **p)
    syn.name = rid
    syn.save(path)
    meta = _base_meta(rid, "synth", method, p, seed, dataset_key, split_cfg, split.sample)
    meta.update(runtime_s=round(time.time() - t0, 2), quality=quality_report(split["train"], syn))
    _write_meta(path, meta)
    return rid, syn, meta


def get_or_create_gnn(dataset_key: str, split_cfg: dict, split: Split, arch: str,
                      params: dict | None, train_params: dict, seed: int, force: bool = False):
    """Returns (release_id, GNNRelease with the train graph attached as server graph, meta)."""
    from graphrecon.releases.gnn import MODELS, TRAIN_DEFAULTS, GNNRelease, train_gnn

    p = MODELS.resolve(arch, params)
    unknown = set(train_params) - set(TRAIN_DEFAULTS)
    if unknown:
        raise KeyError(f"unknown GNN training params {sorted(unknown)}; valid: {sorted(TRAIN_DEFAULTS)}")
    t = {**TRAIN_DEFAULTS, **train_params}
    rid = release_id("gnn", dataset_key, split_cfg, split.sample, arch, {"arch": p, "train": t}, seed)
    path = RELEASES / rid
    if not force and GNNRelease.exists(path) and (path / "meta.json").exists():
        return rid, GNNRelease.load(path, server_graph=split["train"]), read_meta(rid)
    t0 = time.time()
    rel, metrics = train_gnn(split["train"], arch, p, t, seed=seed + 1000 * split.sample,
                             eval_graphs={"holdout": split["holdout"]})
    rel.save(path)
    meta = _base_meta(rid, "gnn", arch, {"arch": p, "train": t}, seed, dataset_key, split_cfg, split.sample)
    meta.update(runtime_s=round(time.time() - t0, 2), utility=metrics)
    _write_meta(path, meta)
    return rid, rel, meta
