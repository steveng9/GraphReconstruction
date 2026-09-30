#!/usr/bin/env bash
# Run every proof-of-concept experiment (idempotent: finished runs are skipped).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p workspace/logs
for cfg in poc_synth_node_amherst poc_synth_node_amherst_anchored poc_synth_edge_amherst \
           poc_gnn_node_amherst poc_gnn_edge_cora poc_gnn_edge_amherst poc_synth_node_jhu; do
  echo "=== $cfg $(date +%T)"
  OMP_NUM_THREADS=4 $PY -m graphrecon run configs/experiments/$cfg.yaml > workspace/logs/$cfg.log 2>&1
  tail -1 workspace/logs/$cfg.log
done
