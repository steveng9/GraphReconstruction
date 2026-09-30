"""Node-attribute reconstruction from a trained GNN (model inversion setting).

The released GNN was trained on the private graph with the sensitive attributes
among its *inputs*; the adversary knows the targets' QI attributes (including the
model's output label, as in Fredrikson et al.) and the target graph, and may hold
auxiliary population data (a disjoint graph) to fit priors.
"""
