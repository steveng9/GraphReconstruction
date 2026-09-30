"""Summary tables from the results DB."""
from __future__ import annotations

import pandas as pd

from graphrecon.results.db import ResultsDB


def pivot(db: ResultsDB, experiment: str, target: str | None = None, metric: str | None = None,
          columns: str = "release_method", with_std: bool = True) -> pd.DataFrame:
    """Rows: attack label (in first-run order). Columns: ``columns`` x target split.
    Cells: mean over samples and seeds (``mean±std`` when there are several)."""
    df = db.query("SELECT * FROM results WHERE experiment=?", (experiment,))
    if df.empty:
        return df
    scope = df["scope"].iloc[0]
    target = target or ("_mean" if scope == "node" else "edges")
    metric = metric or ("r_adv" if scope == "node" else "auc")
    df = df[(df["target"] == target) & (df["metric"] == metric)]
    order = list(dict.fromkeys(df.sort_values("run_id")["attack_label"]))
    col_order = list(dict.fromkeys(df.sort_values("run_id")[columns]))
    g = df.groupby(["attack_label", columns, "target_split"])["value"]
    mean, std, n = g.mean().unstack([1, 2]), g.std().unstack([1, 2]), g.count().unstack([1, 2])
    cols = [c for c in pd.MultiIndex.from_product([col_order, ["train", "holdout"]]) if c in mean.columns]
    mean, std, n = mean.reindex(index=order, columns=cols), std.reindex(index=order, columns=cols), \
        n.reindex(index=order, columns=cols)
    digits = 1 if metric in ("r_adv", "accuracy") else 3
    if not with_std or n.max().max() <= 1:
        return mean.round(digits)
    return mean.round(digits).astype(str) + "±" + std.round(digits).astype(str)
