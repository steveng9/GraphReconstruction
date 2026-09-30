"""SQLite results database.

Three tables:

* ``releases`` -- one row per release (synthetic graph or trained GNN) with its
                  full configuration and quality / utility report;
* ``runs``     -- one row per scored attack run, uniquely keyed by (release,
                  scope, target split, attack label, attack params, task,
                  knowledge, seed), so re-running an experiment skips finished work.
                  A newer run of the same label whose *params* changed (e.g. a new
                  default) marks the older one ``superseded``: kept for audit,
                  excluded from the ``results`` view;
* ``scores``   -- long format (run_id, target, metric, value). ``target`` is a
                  hidden attribute, ``_mean`` (node scope) or ``edges`` (edge scope).

The ``results`` view joins all three for analysis:

    SELECT attack_label, AVG(value) FROM results
    WHERE scope='node' AND target='_mean' AND metric='r_adv' GROUP BY attack_label;
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from graphrecon.config import canonical_json, git_commit, param_hash
from graphrecon.paths import RESULTS_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS releases (
    release_id    TEXT PRIMARY KEY,   -- <kind>/<dataset key>/<split>/s<k>/<method>-<hash>
    kind          TEXT NOT NULL,      -- 'synth' | 'gnn'
    dataset_key   TEXT NOT NULL,      -- dataset name + params hash
    dataset       TEXT NOT NULL,      -- human-readable dataset (e.g. fb100-Amherst41)
    sample        INTEGER NOT NULL,   -- which disjoint node part trained the release
    method        TEXT NOT NULL,      -- generator (synth) or architecture (gnn)
    params_json   TEXT NOT NULL,
    seed          INTEGER NOT NULL,
    report_json   TEXT,               -- synth: quality report; gnn: utility metrics
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment     TEXT NOT NULL,     -- experiment config name
    release_id     TEXT NOT NULL REFERENCES releases(release_id),
    scope          TEXT NOT NULL,     -- 'node' | 'edge' | 'graph'
    target_split   TEXT NOT NULL,     -- 'train' (members) | 'holdout' (non-members)
    attack         TEXT NOT NULL,     -- registered attack name
    attack_label   TEXT NOT NULL,     -- display label (distinguishes parameter variants)
    attack_params_json TEXT NOT NULL, -- fully resolved attack parameters
    attack_hash    TEXT NOT NULL,
    task_json      TEXT NOT NULL,
    task_hash      TEXT NOT NULL,
    knowledge_json TEXT NOT NULL,
    knowledge_hash TEXT NOT NULL,
    seed           INTEGER NOT NULL,
    runtime_s      REAL,
    predictions_path TEXT,            -- saved per-target outputs (npz), if kept
    info_json      TEXT,              -- attack diagnostics (e.g. CoBP-G calibration)
    superseded     INTEGER NOT NULL DEFAULT 0,  -- 1 = replaced by a newer run of the same label
    created_at     TEXT,
    git_commit     TEXT,
    UNIQUE (release_id, scope, target_split, attack_label, attack_hash, task_hash, knowledge_hash, seed)
);
CREATE TABLE IF NOT EXISTS scores (
    run_id  INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    target  TEXT NOT NULL,            -- hidden attribute, '_mean', or 'edges'
    metric  TEXT NOT NULL,            -- r_adv, accuracy, n_scored | auc, ap, precision_at_k, ...
    value   REAL,
    PRIMARY KEY (run_id, target, metric)
);
CREATE VIEW IF NOT EXISTS results AS
SELECT r.run_id, r.experiment, rel.dataset, rel.kind AS release_kind, rel.method AS release_method,
       rel.sample, r.release_id, r.scope, r.target_split, r.attack, r.attack_label, r.seed,
       s.target, s.metric, s.value, r.runtime_s, r.created_at
FROM runs r JOIN releases rel USING (release_id) JOIN scores s USING (run_id)
WHERE r.superseded = 0;
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ResultsDB:
    def __init__(self, path=RESULTS_DB):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=60)
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def upsert_release(self, meta: dict, dataset: str):
        report = meta.get("quality", meta.get("utility"))
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO releases VALUES (?,?,?,?,?,?,?,?,?,?)",
                (meta["release_id"], meta["kind"], meta["dataset_key"], dataset, meta["sample"],
                 meta["method"], canonical_json(meta["params"]), meta["seed"],
                 canonical_json(report) if report is not None else None, meta.get("created_at")))

    @staticmethod
    def run_key(release_id, scope, target_split, label, attack_params, task, knowledge, seed) -> dict:
        return {"release_id": release_id, "scope": scope, "target_split": target_split,
                "attack_label": label, "attack_hash": param_hash(attack_params),
                "task_hash": param_hash(task), "knowledge_hash": param_hash(knowledge), "seed": seed}

    def find_run(self, key: dict) -> int | None:
        where = " AND ".join(f"{k}=?" for k in key) + " AND superseded=0"
        with self.connect() as con:
            row = con.execute(f"SELECT run_id FROM runs WHERE {where}", tuple(key.values())).fetchone()
        return row[0] if row else None

    def delete_run(self, run_id: int):
        with self.connect() as con:
            con.execute("DELETE FROM runs WHERE run_id=?", (run_id,))

    def insert_run(self, key: dict, experiment: str, attack: str, attack_params, task, knowledge,
                   scores: dict, runtime_s: float, predictions_path: str | None = None,
                   info: dict | None = None) -> int:
        with self.connect() as con:
            cur = con.execute(
                "INSERT INTO runs (experiment, release_id, scope, target_split, attack, attack_label, "
                "attack_params_json, attack_hash, task_json, task_hash, knowledge_json, knowledge_hash, "
                "seed, runtime_s, predictions_path, info_json, created_at, git_commit) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (experiment, key["release_id"], key["scope"], key["target_split"], attack,
                 key["attack_label"], canonical_json(attack_params), key["attack_hash"],
                 canonical_json(task), key["task_hash"], canonical_json(knowledge),
                 key["knowledge_hash"], key["seed"], runtime_s, predictions_path,
                 canonical_json(info) if info else None, _now(), git_commit()))
            run_id = cur.lastrowid
            con.execute(
                "UPDATE runs SET superseded=1 WHERE run_id<>? AND release_id=? AND scope=? AND "
                "target_split=? AND attack_label=? AND task_hash=? AND knowledge_hash=? AND seed=?",
                (run_id, key["release_id"], key["scope"], key["target_split"], key["attack_label"],
                 key["task_hash"], key["knowledge_hash"], key["seed"]))
            con.executemany("INSERT INTO scores VALUES (?,?,?,?)",
                            [(run_id, t, m, v) for (t, m), v in scores.items()])
        return run_id

    def query(self, sql: str, params=()) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query(sql, con, params=params)
