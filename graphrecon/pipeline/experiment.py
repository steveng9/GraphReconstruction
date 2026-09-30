"""Config-driven experiment runner.

    dataset -> node split (per sample) -> release (cached) -> target view (per target
    split, seed) -> each attack -> score -> results DB (+ optional saved predictions)

A run already in the DB (same release, scope, target split, attack label and
params, task, knowledge, seed) is skipped unless ``force``; so re-running a config
after adding an attack or a sample only computes what is missing. Every attack
sees the identical view for a given (sample, target split, seed).

See configs/experiments/*.yaml for the schema.
"""
from __future__ import annotations

import time
import traceback
import zlib

import numpy as np

from graphrecon.attacks import ATTACKS
from graphrecon.attacks.base import GNNAccess
from graphrecon.config import load_yaml, param_hash
from graphrecon.data.datasets import dataset_key, load_dataset
from graphrecon.data.splits import SPLIT_DEFAULTS, make_split
from graphrecon.paths import ATTACK_OUTPUTS
from graphrecon.releases.store import get_or_create_gnn, get_or_create_synth
from graphrecon.results.db import ResultsDB
from graphrecon.scoring import score_edges, score_nodes
from graphrecon.threat.views import (EDGE_KNOWLEDGE_DEFAULTS, NODE_KNOWLEDGE_DEFAULTS, EdgeTask,
                                     NodeTask, make_edge_view, make_node_view)

TOP_LEVEL_KEYS = {"name", "description", "dataset", "split", "samples", "release", "task", "knowledge",
                  "adversary", "target_splits", "attacks", "seeds", "save_predictions"}


def _stable(s: str) -> int:
    return zlib.crc32(s.encode())


def _validate(cfg: dict):
    unknown = set(cfg) - TOP_LEVEL_KEYS
    if unknown:
        raise KeyError(f"unknown experiment keys {sorted(unknown)}; valid: {sorted(TOP_LEVEL_KEYS)}")
    kind, scope = cfg["release"]["kind"], cfg["task"]["scope"]
    if scope == "graph":
        raise NotImplementedError("graph-level scope is scaffolded but has no attacks yet")
    labels = [a.get("label", a["name"]) for a in cfg["attacks"]]
    if len(labels) != len(set(labels)):
        raise ValueError("attack labels must be unique; give parameter variants a 'label'")
    for a in cfg["attacks"]:                            # fail before any work on a bad attack config
        ATTACKS.resolve((kind, scope, a["name"]), a.get("params"))


def _task(cfg):
    t = dict(cfg["task"])
    scope = t.pop("scope")
    return scope, (NodeTask(**t) if scope == "node" else EdgeTask(**t))


def run_experiment(cfg: dict | str, force: bool = False, only_attacks: list[str] | None = None,
                   samples: list[int] | None = None, db: ResultsDB | None = None, verbose: bool = True):
    if not isinstance(cfg, dict):
        cfg = load_yaml(cfg)
    _validate(cfg)
    db = db or ResultsDB()
    log = print if verbose else (lambda *a, **k: None)

    ds = cfg["dataset"]
    graph = load_dataset(ds["name"], ds.get("params"))
    dkey = dataset_key(ds["name"], ds.get("params"))
    split_cfg = {**SPLIT_DEFAULTS, **cfg.get("split", {})}
    rel_cfg = cfg["release"]
    kind = rel_cfg["kind"]
    scope, task = _task(cfg)
    task_dict = {"scope": scope, **{k: v for k, v in cfg["task"].items() if k != "scope"}}
    kdef = NODE_KNOWLEDGE_DEFAULTS if scope == "node" else EDGE_KNOWLEDGE_DEFAULTS
    knowledge = {**kdef, **cfg.get("knowledge", {})}
    grant_aux = cfg.get("adversary", {}).get("aux", kind == "gnn")
    target_splits = cfg.get("target_splits", ["train"])
    seeds = cfg.get("seeds", [0])
    attacks = [a for a in cfg["attacks"] if not only_attacks or a.get("label", a["name"]) in only_attacks]
    log(f"[{cfg['name']}] {graph.summary()}")

    n_done = n_skip = n_fail = 0
    for sample in samples if samples is not None else cfg.get("samples", [0]):
        split = make_split(graph, sample, **split_cfg)
        for m in rel_cfg["methods"]:
            method, mparams = m["name"], m.get("params", {})
            t0 = time.time()
            if kind == "synth":
                rid, rel, meta = get_or_create_synth(dkey, split_cfg, split, method, mparams,
                                                     rel_cfg.get("seed", 0))
                access = rel
            else:
                rid, rel, meta = get_or_create_gnn(dkey, split_cfg, split, method, mparams,
                                                   rel_cfg.get("train", {}), rel_cfg.get("seed", 0))
                access = GNNAccess(rel, split["aux"] if grant_aux else None)
            db.upsert_release(meta, graph.name)
            log(f"  release {rid} ({time.time() - t0:.1f}s)")
            for ts in target_splits:
                if kind == "gnn" and scope == "edge" and ts != "train":
                    log(f"    skip target split {ts!r}: GNN edge attacks query the training graph")
                    continue
                target = split[ts]
                for seed in seeds:
                    view_seed = [seed, sample, _stable(ts)]
                    if scope == "node":
                        view, truth = make_node_view(target, task, knowledge, np.random.default_rng(view_seed).integers(2**31))
                    else:
                        view, truth = make_edge_view(target, task, knowledge, np.random.default_rng(view_seed).integers(2**31))
                    for a in attacks:
                        name, label = a["name"], a.get("label", a["name"])
                        params = ATTACKS.resolve((kind, scope, name), a.get("params"))
                        key = db.run_key(rid, scope, ts, label, {"name": name, **params}, task_dict, knowledge, seed)
                        existing = db.find_run(key)
                        if existing is not None:
                            if not force:
                                n_skip += 1
                                continue
                            db.delete_run(existing)
                        rng = np.random.default_rng([seed, sample, _stable(label)])
                        t0 = time.time()
                        try:
                            out = ATTACKS.get((kind, scope, name)).fn(access, view, rng, **params)
                        except Exception as e:
                            n_fail += 1
                            log(f"    FAIL {label} [{ts}]: {type(e).__name__}: {e}")
                            if verbose:
                                traceback.print_exc()
                            continue
                        runtime = time.time() - t0
                        pred_path, info = None, None
                        if scope == "node":
                            preds = out.predictions(view)
                            scores, rows = score_nodes(truth, preds, view.known)
                            info = out.info
                            headline = f"R_adv={scores[('_mean', 'r_adv')]:.1f}  " + " ".join(
                                f"{h}={scores[(h, 'r_adv')]:.1f}" for h in task.hidden)
                        else:
                            scores = score_edges(truth, out)
                            headline = (f"AUC={scores.get(('edges', 'auc'), float('nan')):.3f} "
                                        f"AP={scores.get(('edges', 'ap'), float('nan')):.3f}")
                        if cfg.get("save_predictions", False):
                            path = ATTACK_OUTPUTS / rid / scope / ts / f"{label}-{param_hash(key)}.npz"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            if scope == "node":
                                np.savez_compressed(path, row_scores=rows,
                                                    **{f"proba__{h}": P.astype(np.float32) for h, P in out.probas.items()},
                                                    **{f"truth__{h}": t for h, t in truth.items()})
                            else:
                                np.savez_compressed(path, pairs=view.pairs, truth=truth, scores=out)
                            pred_path = str(path.relative_to(ATTACK_OUTPUTS))
                        db.insert_run(key, cfg["name"], name, params, task_dict, knowledge, scores,
                                      runtime, pred_path, info)
                        n_done += 1
                        log(f"    s{sample} {method:10s} {ts:8s} {label:22s} {headline}  ({runtime:.1f}s)")
    log(f"[{cfg['name']}] done: {n_done} run, {n_skip} skipped (already in DB), {n_fail} failed")
    return n_done, n_skip, n_fail
