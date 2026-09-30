"""Synthetic graph generators: ``fn(graph, rng, **params) -> AttributedGraph``.

A generator sees the private (train) graph, including every attribute, and returns
a synthetic graph with the same attribute schema.
"""
from graphrecon.registry import Registry

GENERATORS = Registry("generator")

from graphrecon.releases.synth import generators  # noqa: E402,F401
