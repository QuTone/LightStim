#!/usr/bin/env bash
# Start tmux sessions for TG and LS full-noise distillation sweeps.
# TG uses pre-cached circuits (circuits/TG_7to1_d{d}_r1.stim).
# LS builds and caches circuits on first run.
#
# Usage (from repo root):
#   bash eval/logical_circuit_benchmark/distillation/start_distill_sweep.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DIST_DIR="$REPO/eval/logical_circuit_benchmark/distillation"
PYTHON="$REPO/venv/bin/python"

start_session() {
    local name="$1" script="$2" logfile="$3"
    if tmux has-session -t "$name" 2>/dev/null; then
        echo "Skip: session '$name' already exists  (tmux attach -t $name)"
        return
    fi
    tmux new-session -d -s "$name" bash -lc \
        "cd '$REPO' && $PYTHON $script 2>&1 | tee $logfile"
    echo "Started: $name  →  $logfile"
}

mkdir -p "$DIST_DIR/tg_7to1/results" "$DIST_DIR/ls_7to1/results"

# TG: d=3,5,7  p=1e-4,3e-4,1e-3,3e-3,1e-2  (Option A)
start_session distill-tg \
    "$DIST_DIR/tg_7to1/run_tg_full_sweep.py \
        -d 3 5 7 \
        --p-values 1e-4 3e-4 1e-3 3e-3 1e-2 \
        --num-workers 8 --max-shots 100000000 --max-errors 100" \
    "$DIST_DIR/tg_7to1/results/run_tg_full_sweep.log"

# LS: d=3,5,7  p=logspace(-5,-1,6)  (recover original range)
start_session distill-ls \
    "$DIST_DIR/ls_7to1/run_ls_full_sweep.py \
        -d 3 5 7 \
        --p-values 1e-5 6.31e-5 3.98e-4 2.51e-3 1.58e-2 1e-1 \
        --num-workers 8 --max-shots 100000000 --max-errors 100" \
    "$DIST_DIR/ls_7to1/results/run_ls_full_sweep.log"

echo ""
echo "Sessions:"
tmux ls 2>/dev/null | grep distill || true
echo ""
echo "Attach:  tmux attach -t distill-tg"
echo "         tmux attach -t distill-ls"
echo "CSV out: $DIST_DIR/tg_7to1/TG_full_noise_results.csv"
echo "         $DIST_DIR/ls_7to1/LS_full_noise_results.csv"
