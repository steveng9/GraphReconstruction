# Graph-level scope (not yet implemented)

Reserved for attacks that reconstruct graph-level or subgraph-level information.
See `notes/DESIGN.md` §1 "Graph scope" for the two candidate formulations
(graph-level records vs. ego-network reconstruction). Register attacks here with
`register_attack(<kind>, "graph", name, defaults)` once the task, view and scorer
exist in `threat/views.py` and `scoring/`.
