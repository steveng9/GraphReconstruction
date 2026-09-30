# Design notes

Working document: threat models, design decisions, open questions. Update it when a decision changes.

## 1. Threat models

The private object is an attributed graph G = (V, E, X). The defender releases
R(G) as either a **synthetic graph** or a **trained GNN**. The adversary targets a
set of real nodes, knows part of their data (a *view*) and must reconstruct the
rest.

### Node scope: attribute reconstruction (implemented)

| | synthetic graph release | GNN release |
|---|---|---|
| adversary knows | QI attributes of targets; observed fraction of target edges; optionally a fraction of targets' hidden values (anchors) | the same, and the model's output label for targets (Fredrikson-style); black-box inductive query access; optional auxiliary population graph |
| learns from | the synthetic graph (fully labelled) | the model (queries) + auxiliary data |
| references | Mode, Random | Mode (aux), Imputation (aux; no model) |

The memorization test works as in the SoK: members are nodes of the train part
the release was built from; non-members are a disjoint holdout part of the same
graph. Their gap separates memorization from distributional inference.

### Edge scope: link reconstruction (implemented)

The adversary knows target attributes (`known_attrs`) and a fraction of the
target edges. It scores every unobserved true edge against as many sampled
non-edges (AUC / AP / precision@k). For GNN releases, link stealing queries the
model served on its private training graph (members only).

### Graph scope (not yet implemented)

Two candidate formulations; we should pick one before building anything:

1. **Graph-level records** (molecule / TU datasets): each graph is a record.
   A synthetic *set of graphs* or a graph classifier is released; the adversary
   knows a target graph's structure (or part of it) and infers a hidden
   graph-level property or label. This is the closest analogue of the tabular
   SoK, but needs conditional graph generators (e.g. DiGress, GDSS) we have not
   vetted.
2. **Subgraph / ego-network reconstruction** in one large graph: from the release
   and a target's partial ego network, reconstruct the rest of its k-hop
   neighbourhood (edges + attributes jointly). This combines node and edge scope
   and is where joint inference (BP over both) could be novel.

## 2. Decisions

- **Release kind at the top, scope below**: the release decides the attack's
  input type and the pipeline that produces it; the scope decides the scoring.
  Both are registry keys and DB columns, so either pivot is one `GROUP BY`.
- **PyG, not DGL**: DGL development has largely stalled; PyG is maintained and
  loads OGB directly. torch 2.4 keeps DGL 2.4 installable if ever needed.
- **Facebook100** as the main node-attribute dataset: real demographic
  categorical attributes (status, gender, major, dorm, year), the graph analogue
  of a census table. Cardinality is capped (top-11 + "other"); minor (61% missing)
  and high school (~2k values) are dropped. Planetoid/Cora serves the GNN link
  experiments (the standard link-stealing benchmark).
- **Random node partitions** give disjoint samples. Induced subgraphs thin
  degrees by ~1/n_parts (Amherst: mean degree 81 → ~18 with 4 parts). This is a
  real limitation: targets see only part of their friends.
- **Generators chosen for confidence, spanning structure-attribute coupling**:
  Identity (full coupling, the ceiling), ChungLu (none, the null), AGM
  (per-attribute homophily), DCSBM (community-mediated). DP variants (e.g.
  AGM-DP, Jorgensen et al. 2016) and deep generators (GraphMaker, etc.) come later.
- **Rarity-weighted R_adv** kept verbatim from the SoK so numbers are comparable.

## 3. CoBP-G

Classifier unaries + loopy BP over a pairwise MRF:

- graph factors ψ_h(h_i, h_j) on each observed target edge (homophily);
- row factors ψ_hg(h_i, g_i) on a Chow-Liu tree of hidden attributes (as in CoBP-RA).

**Residual potentials** ψ = log (O + α) / (E + α): O counts co-occurrences over
synthetic pairs, E sums the out-of-fold classifier posteriors q_i(v) q_j(w) over
the same pairs. This is the global analogue of CoBP-RA's neighbour-local PMI,
which exists to avoid double counting evidence the unary already has.

**Self-calibration**: rehearse on the synthetic graph (out-of-fold unaries, same
observed-edge and anchor fractions), with potentials cross-fitted over two node
folds so the rehearsal cannot fit its own noise. Adopt the best (graph, row)
strength only if its paired per-node gain over no coupling reaches `calib_z` SE.

History: the first version calibrated in-sample. On the null ChungLu release it
"found" gains of z≈1.3 and hurt targets (dorm R_adv 16 → 10). The node cross-fit
fixed that (null z≈0.4). The cost is conservatism, since each fold's potentials
use ~1/4 of the edges; on DCSBM it now declines BP where BP helped the targets.

## 4. Open questions / ideas

- **Anchors change everything.** With 30% of targets' hidden values public, the
  model-free wvRN (NeighborVote) beats every synthetic-data attack on Amherst.
  So the *graph the adversary already has* can matter more than the release. A
  fair paper must report that baseline, and CoBP-G must use anchors at least as
  well as wvRN does. Next: add a `neighbor_known` feature block (rehearsed on
  synth) to the classifiers and check CoBP-G with anchors in its rehearsal.
- **Calibration bias/variance**: edge-fold cross-fit with a density correction,
  or shrinkage of potential tables toward 0 by their standard error.
- **Structural fingerprinting**: against Identity (naive anonymisation), node
  classifiers reach ~99.6 R_adv on members vs ~25 on non-members. The member
  score is re-identification by degree/clustering/neighbour-QI profile, not
  inference (cf. Backstrom et al. 2007; Narayanan & Shmatikov 2009). It is the
  graph analogue of the SoK's RankSwap/CellSuppression result, and worth its own
  analysis (which structural features identify?).
- **Signal on fb100 without anchors is small**: status + year says little about
  gender/major. Consider richer QI (dorm as QI, major hidden), bigger graphs
  (Penn94) or fewer parts (less degree thinning).
- **GNN side**: model inversion marginalises neighbours' unknown inputs by Monte
  Carlo. A joint version (BP / coordinate ascent over all targets' hidden inputs,
  scored by model likelihood) is the natural "CoBP for GNNs".
- Cross-attribute graph factors (h_i with g_j across an edge); max-product
  decoding; learning potentials by pseudo-likelihood instead of moment matching.

## 5. Related work pointers (to read / cite)

- Attribute inference in social networks: Zheleva & Getoor 2009; Gong & Liu 2016;
  AttriInfer (Jia et al., WWW 2017; MRF + LBP, the closest prior to CoBP-G).
- Collective classification: Sen et al. 2008; wvRN (Macskassy & Provost 2007);
  compatibility-matrix BP (Gatterbauer et al. 2015; ZooBP, Eswaran et al. 2017).
- GNN privacy: link stealing (He et al. 2021); GraphMI (Zhang et al. 2021);
  attribute inference from embeddings (Duddu et al. 2020); "attribute inference
  is just imputation?" (Jayaraman & Evans 2022).
- Graph synthesis with privacy: AGM (Pfeiffer et al. 2014); AGM-DP (Jorgensen et
  al. 2016); DP graph generation surveys; GraphMaker (Li et al. 2024).
