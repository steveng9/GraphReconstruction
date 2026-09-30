"""Isolate tests from the real workspace: point GRAPHRECON_WORKSPACE at a temp dir
*before* graphrecon is imported (paths are resolved at import time)."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="graphrecon-test-")
os.environ["GRAPHRECON_WORKSPACE"] = _tmp
os.environ["GRAPHRECON_DB"] = os.path.join(_tmp, "results.db")
