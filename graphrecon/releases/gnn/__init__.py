"""Trained-GNN releases: a node classifier trained on the private graph.

``MODELS`` maps architecture names to constructors; ``train_gnn`` fits one;
``GNNRelease`` is the published artefact with black-box and white-box access.
"""
from graphrecon.releases.gnn.models import MODELS  # noqa: F401
from graphrecon.releases.gnn.release import GNNRelease  # noqa: F401
from graphrecon.releases.gnn.train import TRAIN_DEFAULTS, train_gnn  # noqa: F401
