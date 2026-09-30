"""End to end on the toy graph: every registered attack runs through the pipeline,
the DB is idempotent, and CoBP-G exploits homophily on the real graph."""
from graphrecon.attacks import list_attacks
from graphrecon.pipeline import run_experiment
from graphrecon.results import ResultsDB


def _cfg(name, kind, scope, attacks, task, **extra):
    methods = [{"name": "Identity"}] if kind == "synth" else [{"name": "SAGE", "params": {"hidden": 16}}]
    cfg = {"name": name, "dataset": {"name": "toy_sbm"}, "split": {"n_parts": 3, "seed": 0},
           "samples": [0], "release": {"kind": kind, "methods": methods}, "task": {"scope": scope, **task},
           "target_splits": ["train"], "attacks": [{"name": a} for a in attacks]}
    if kind == "gnn":
        cfg["release"]["train"] = {"label": "qi", "epochs": 30}
    cfg.update(extra)
    return cfg


def test_every_attack_runs_and_db_is_idempotent():
    db = ResultsDB()
    node = {"qi": ["qi"], "hidden": ["label", "noise"]}
    edge = {"known_attrs": ["qi", "label"]}
    gnn_node = {"qi": ["qi"], "hidden": ["label", "noise"]}
    cases = [("synth", "node", node, {}), ("synth", "edge", edge, {"knowledge": {"observe_edges_frac": 0.5}}),
             ("gnn", "node", gnn_node, {}), ("gnn", "edge", edge, {})]
    for kind, scope, task, extra in cases:
        attacks = [k[2] for k in list_attacks(kind, scope)]
        if (kind, scope) == ("gnn", "node"):
            extra = {**extra, "adversary": {"aux": True}}
        cfg = _cfg(f"t_{kind}_{scope}", kind, scope, attacks, task, **extra)
        done, skipped, failed = run_experiment(cfg, db=db, verbose=False)
        assert failed == 0 and done == len(attacks)
        assert run_experiment(cfg, db=db, verbose=False) == (0, len(attacks), 0)


def test_cobp_beats_its_unary_on_homophilous_graph():
    db = ResultsDB()
    cfg = _cfg("t_cobp", "synth", "node", [], {"qi": ["qi", "noise"], "hidden": ["label"]},
               target_splits=["holdout"])
    cfg["attacks"] = [{"name": "RandomForest"}, {"name": "CoBP-G"}]
    run_experiment(cfg, db=db, verbose=False)
    r = db.query("SELECT attack_label, value FROM results WHERE experiment='t_cobp' "
                 "AND target='_mean' AND metric='r_adv'").set_index("attack_label")["value"]
    assert r["CoBP-G"] > r["RandomForest"] + 2.0
