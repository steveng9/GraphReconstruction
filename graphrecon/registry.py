"""Name -> component registries with declared default parameters.

Every registered component (dataset, generator, GNN model, attack) declares its
parameters and their defaults at registration time. ``resolve`` merges user
parameters over the defaults and rejects unknown keys, so a typo in a config
fails loudly instead of silently running the default.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Entry:
    name: str
    fn: Callable
    defaults: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    doc: str = ""


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[Any, Entry] = {}

    def register(self, key, defaults: dict | None = None, **meta):
        def deco(fn):
            if key in self._entries:
                raise KeyError(f"{self.kind} {key!r} registered twice")
            doc = (fn.__doc__ or "").strip().split("\n\n")[0]
            self._entries[key] = Entry(str(key), fn, dict(defaults or {}), meta, doc)
            return fn
        return deco

    def get(self, key) -> Entry:
        if key not in self._entries:
            raise KeyError(f"unknown {self.kind} {key!r}; available: {sorted(map(str, self._entries))}")
        return self._entries[key]

    def resolve(self, key, params: dict | None) -> dict:
        """Defaults overlaid with ``params``; unknown keys raise."""
        entry = self.get(key)
        params = dict(params or {})
        unknown = set(params) - set(entry.defaults)
        if unknown:
            raise KeyError(f"{self.kind} {key!r} got unknown params {sorted(unknown)}; "
                           f"valid: {sorted(entry.defaults)}")
        out = copy.deepcopy(entry.defaults)
        out.update(params)
        return out

    def keys(self):
        return list(self._entries)

    def items(self):
        return list(self._entries.items())

    def __contains__(self, key):
        return key in self._entries
