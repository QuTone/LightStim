#!/bin/bash
# Full TG distillation sweep: d=[3,5,7], p=[1e-3,1e-4], 3 decoders
# Run inside tmux: tmux new -s tg_sweep 'bash eval/TG_distillation/run_sweep.sh'

set -e
cd /home/xiang/workspace/LightStim
PYTHON=/home/xiang/workspace/LightStim/venv/bin/python
SCRIPT=eval/TG_distillation/TG_distillation_7_to_1.py
OUTDIR=eval/TG_distillation

echo "=========================================="
echo "TG Distillation Sweep — $(date)"
echo "=========================================="

# --- CPU BPOSD ---
echo ""
echo ">>> CPU BPOSD d=3,5,7 p=1e-3,1e-4"
PYTHONUNBUFFERED=1 $PYTHON $SCRIPT \
    -d 3 5 7 -p 1e-3 1e-4 \
    --decoder bposd --backend cpu --num-workers 32 \
    --max-errors 200 --batch-size 50000 -r 1 \
    2>&1 | tee $OUTDIR/sweep_cpu_bposd.log

# Rename results file
mv -f $OUTDIR/TG_distillation_7_to_1_results.json \
      $OUTDIR/results_cpu_bposd.json 2>/dev/null || true

# --- GPU BPOSD ---
echo ""
echo ">>> GPU BPOSD d=3,5,7 p=1e-3,1e-4"
PYTHONUNBUFFERED=1 $PYTHON $SCRIPT \
    -d 3 5 7 -p 1e-3 1e-4 \
    --decoder bposd --backend gpu --num-workers 1 \
    --max-errors 200 --batch-size 10000 -r 1 \
    2>&1 | tee $OUTDIR/sweep_gpu_bposd.log

mv -f $OUTDIR/TG_distillation_7_to_1_results.json \
      $OUTDIR/results_gpu_bposd.json 2>/dev/null || true

# --- MWPF ---
echo ""
echo ">>> MWPF d=3,5,7 p=1e-3,1e-4"
PYTHONUNBUFFERED=1 $PYTHON $SCRIPT \
    -d 3 5 7 -p 1e-3 1e-4 \
    --decoder mwpf --backend cpu --num-workers 32 \
    --max-errors 200 --batch-size 50000 -r 1 \
    2>&1 | tee $OUTDIR/sweep_mwpf.log

mv -f $OUTDIR/TG_distillation_7_to_1_results.json \
      $OUTDIR/results_mwpf.json 2>/dev/null || true

echo ""
echo "=========================================="
echo "All sweeps done — $(date)"
echo "=========================================="
