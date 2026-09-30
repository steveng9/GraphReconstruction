"""Command line entry point: ``python -m graphrecon <command>``.

    run CONFIG [--force] [--attacks L ...] [--samples K ...]   run an experiment config
    list [datasets|generators|models|attacks]                   registered components + defaults
    results EXPERIMENT [--target T] [--metric M] [--by COL]     summary table from the DB
    releases [--kind synth|gnn]                                 releases in the DB with their reports
"""
from __future__ import annotations

import argparse
import json

import pandas as pd


def _cmd_run(a):
    from graphrecon.pipeline import run_experiment
    run_experiment(a.config, force=a.force, only_attacks=a.attacks, samples=a.samples)


def _cmd_list(a):
    from graphrecon.attacks import ATTACKS
    from graphrecon.data.datasets import DATASETS
    from graphrecon.releases.gnn import MODELS
    from graphrecon.releases.synth import GENERATORS
    regs = {"datasets": DATASETS, "generators": GENERATORS, "models": MODELS, "attacks": ATTACKS}
    for name, reg in regs.items():
        if a.what and a.what != name:
            continue
        print(f"== {name}")
        for key, e in sorted(reg.items(), key=lambda kv: str(kv[0])):
            label = "/".join(key) if isinstance(key, tuple) else key
            print(f"  {label:34s} {e.doc}")
            if a.verbose and e.defaults:
                print(f"  {'':34s} defaults: {json.dumps(e.defaults)}")


def _cmd_results(a):
    from graphrecon.results import ResultsDB
    from graphrecon.results.tables import pivot
    pd.set_option("display.width", 250, "display.max_columns", 50)
    t = pivot(ResultsDB(), a.experiment, a.target, a.metric, a.by)
    print(t.to_string() if not t.empty else f"no results for experiment {a.experiment!r}")


def _cmd_releases(a):
    from graphrecon.results import ResultsDB
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    q = "SELECT release_id, kind, dataset, sample, method, report_json FROM releases"
    df = ResultsDB().query(q + (" WHERE kind=?" if a.kind else ""), (a.kind,) if a.kind else ())
    for _, r in df.iterrows():
        rep = json.loads(r["report_json"] or "{}")
        if r["kind"] == "synth":
            hom = {k: [round(x, 2) for x in v["edge_homophily"]] for k, v in rep.get("attributes", {}).items()}
            print(f"{r['release_id']}\n   edges={rep.get('n_edges')} homophily(real,syn)={hom}")
        else:
            print(f"{r['release_id']}\n   {rep}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="graphrecon", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("config"); r.add_argument("--force", action="store_true")
    r.add_argument("--attacks", nargs="*"); r.add_argument("--samples", nargs="*", type=int)
    r.set_defaults(fn=_cmd_run)
    ls = sub.add_parser("list"); ls.add_argument("what", nargs="?"); ls.add_argument("-v", "--verbose", action="store_true")
    ls.set_defaults(fn=_cmd_list)
    rs = sub.add_parser("results"); rs.add_argument("experiment"); rs.add_argument("--target")
    rs.add_argument("--metric"); rs.add_argument("--by", default="release_method")
    rs.set_defaults(fn=_cmd_results)
    rl = sub.add_parser("releases"); rl.add_argument("--kind"); rl.set_defaults(fn=_cmd_releases)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
