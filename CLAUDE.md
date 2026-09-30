# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

Reconstruction attacks (attribute / link / graph-level inference) against trained
GNNs and synthetic graph data. It is the graph follow-up to the tabular SoK in
`../Reconstruction` (PoPETs 2027; CoBP-RA lives in
`../Reconstruction/attacks/marginal_rf.py`). The goal is a methods paper; the
paper source is in `paper/` (edit `content.tex`, `appendix.tex`, `notes.tex`).

## Environment and commands

```bash
conda activate graphrecon_        # python 3.11, torch 2.4.1+cu121, torch_geometric 2.6.1
pytest                            # isolated temp workspace; ~15 s
python -m graphrecon list -v      # registered components and their defaults
python -m graphrecon run configs/experiments/<name>.yaml [--force] [--attacks LABEL ...] [--samples K ...]
python -m graphrecon results <experiment> [--target ATTR] [--metric r_adv|accuracy|auc|ap]
```

For experiments that must not touch the main DB, set `GRAPHRECON_DB=workspace/<scratch>.db`.

## Architecture (read before editing)

- **Registries with defaults** (`graphrecon/registry.py`): datasets, generators,
  GNN models and attacks declare their parameters at registration; configs may
  only override declared keys. Add a parameter by adding it to `defaults`.
- **Attack registry key** = (release kind, scope, name), with kind in {synth, gnn}
  and scope in {node, edge, graph}. Signatures: node attacks return
  `NodePrediction` (per-attribute (n, C) probabilities over the *full* category
  set); edge attacks return a score per candidate pair.
- **Views vs. truth** (`threat/views.py`): attacks receive a view containing only
  what the threat model grants; the truth goes to the scorer alone. Never give an
  attack the target graph itself.
- **AttributedGraph** (`data/graph.py`): undirected edges stored once (u < v),
  categorical attributes as int codes with MISSING = -1, a schema of category
  labels, and optional dense `x`.
- **Splits** (`data/splits.py`): random node partition into `n_parts`; sample k
  uses part k (train / members), k+1 (holdout / non-members), k+2 (aux).
- **Releases are cached** by content hash in `workspace/releases/<release id>/`
  with `meta.json`. Changing a generator's code does *not* change its hash:
  delete the cached directory (or bump a param) after changing generator logic.
- **Results DB** (`results/db.py`): a run is keyed by release, scope, split,
  label, param hash, task, knowledge and seed. A new run of the same label with
  changed params supersedes the old one (`superseded = 1`, hidden from the
  `results` view).

## Conventions

- Scores: node R_adv on a 0-100 scale (mode baseline = 100/C); edge AUC/AP.
  Targets whose hidden value is missing or known to the adversary are not scored.
- Every attack must be deterministic given `rng`; the runner seeds it from
  (seed, sample, label).
- Keep generated data out of git (`workspace/`). The disk is nearly full (~70 GB
  free as of 2026-09): prefer small graphs, and do not cache large OGB datasets
  casually.
- Design rationale, threat models and open questions: `notes/DESIGN.md`. Update
  it when a design decision changes.
