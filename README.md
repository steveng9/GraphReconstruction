# GraphReconstruction

Reconstruction attacks (attribute inference, link inference, graph-level
inference) against **trained GNNs** and **synthetic graph data**. It is the graph
counterpart of the tabular reconstruction SoK (`../Reconstruction`, PoPETs 2027):
the same scoring (rarity-weighted R_adv), memorization test (member vs.
non-member targets) and registry / config / results-DB discipline, carried over
to network data.

## Setup

```bash
conda env create -f environment.yaml      # creates graphrecon_ (torch 2.4 + PyG 2.6 + OGB), installs this package
conda activate graphrecon_
pytest                                     # ~15 s, runs in an isolated temp workspace
```

## Quickstart

```bash
python -m graphrecon list -v                                     # every dataset / generator / model / attack + defaults
python -m graphrecon run configs/experiments/poc_synth_node_amherst.yaml
python -m graphrecon results poc_synth_node_amherst              # attacks x releases x {members, non-members}
python -m graphrecon results poc_synth_node_amherst --target dorm --metric accuracy
python -m graphrecon releases --kind synth                       # generator fidelity (edge homophily real vs. synthetic)
scripts/run_poc.sh                                               # every proof-of-concept experiment
```

Runs are idempotent: re-running a config computes only what is missing from the DB.

## How it is organised

The top level splits by **what is released** (it decides the attack's input and
the pipeline that produces it); below that, by **what is reconstructed** (it
decides the scoring). Both are keys in the attack registry and columns in the
DB, so results pivot either way.

```
graphrecon/
  data/            AttributedGraph container, dataset loaders, node splits, feature builders
  releases/
    synth/         synthetic graph generators (Identity, ChungLu, AGM, DCSBM) + fidelity report
    gnn/           GNN architectures, training, and the released-model API (black-/white-box)
    store.py       content-addressed release cache (workspace/releases/<release id>/)
  threat/          what the adversary knows (views) vs. the truth (kept for the scorer only)
  attacks/
    common/        classifier factory, loopy belief propagation engine
    synth/{node,edge,graph}/
    gnn/{node,edge,graph}/
  scoring/         node: R_adv, accuracy; edge: AUC, AP, precision@k
  results/         SQLite results DB + summary tables
  pipeline/        config-driven experiment runner
configs/experiments/  one YAML per experiment
tests/             pytest suite
notes/DESIGN.md    threat models, design decisions, open questions, roadmap
paper/             the Overleaf project (NeurIPS / ACM wrappers around one body of text)
workspace/         (gitignored) datasets, releases, attack outputs, results.db
```

### Pipeline

```
dataset ─► node split (sample k: train / holdout / aux parts)
            └─► release (generator or GNN trained on the train part; cached by config hash)
                 └─► target view (members = train part, non-members = holdout part)
                      └─► attack ─► score ─► results.db (+ per-target predictions)
```

### Attacks implemented (v0.1)

| release | scope | attacks |
|---|---|---|
| synth | node | `Mode`, `Random` (reference points); `RandomForest`, `LightGBM`, `LogisticRegression`, `MLP`, `NaiveBayes`, `KNN` on tabularised node features; `GNNClassifier`; `NeighborVote` (wvRN); **`CoBP-G`** |
| synth | edge | `Random`, `AttributeSimilarity`, `CommonNeighbors`, `AdamicAdar`, `Jaccard`, `PreferentialAttachment`, `PairClassifier` |
| gnn | node | `Mode`, `Imputation` (no model access; the control), `ModelInversion` (Monte-Carlo black-box, with uniform / marginal / imputation priors) |
| gnn | edge | `LinkStealing` (He et al. 2021, attack-0), plus the release-independent edge baselines |
| * | graph | scaffolded, not yet implemented (see `notes/DESIGN.md`) |

**CoBP-G** (`attacks/synth/node/cobp.py`) is the graph analogue of CoBP-RA. It
uses classifier posteriors as unaries and runs loopy BP over a pairwise MRF with
factors *across edges* (same attribute on neighbouring targets) and *across the
row* (hidden attributes within a node). Potentials are residual PMI estimated on
the synthetic graph, and factor strengths are self-calibrated by rehearsing the
attack on the synthetic graph.

## Extending

Every component registers itself with its default parameters; unknown config
keys raise.

```python
@GENERATORS.register("MyGen", defaults={"p": 0.1})                    # releases/synth/
def my_gen(graph, rng, p): ...                                         # -> AttributedGraph

@register_attack("synth", "node", "MyAttack", defaults={"k": 5})      # attacks/synth/node/
def my_attack(release, view, rng, k): ...                             # -> NodePrediction
```

Then import the module in the package `__init__` and reference it by name in a config.

## Results database

`workspace/results.db` (SQLite). Tables `releases`, `runs`, `scores` (long
format), and a `results` view that joins them and excludes superseded runs.
Schema comments are in `graphrecon/results/db.py`.

```sql
SELECT release_method, target_split, attack_label, AVG(value)
FROM results WHERE experiment='poc_synth_node_amherst' AND target='_mean' AND metric='r_adv'
GROUP BY 1, 2, 3;
```
