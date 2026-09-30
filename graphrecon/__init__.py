"""graphrecon: reconstruction attacks against GNNs and synthetic graph data.

Layout
------
data/       attributed-graph container, dataset loaders, node splits, feature builders
releases/   what the defender publishes: synthetic graphs (synth/) or trained GNNs (gnn/)
threat/     what the adversary knows about the targets (views) and what it must infer (tasks)
attacks/    attacks, organised as attacks/<release kind>/<scope>/ with scope in {node, edge, graph}
scoring/    metrics per scope
results/    SQLite results database
pipeline/   config-driven experiment runner
"""
