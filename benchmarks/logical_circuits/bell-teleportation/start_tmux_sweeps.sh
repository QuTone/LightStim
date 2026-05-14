#!/usr/bin/env bash
# Start three detached tmux sessions for TG / ZZ-LS / XX-LS sweeps.
# Uses paths derived from this script — no REPO/BT env vars required.
set -euo pipefail

BT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$BT/../../.." && pwd)"
mkdir -p "$BT/results"

if [[ -f "$REPO/venv/bin/activate" ]]; then
  INNER="source '$REPO/venv/bin/activate' && cd '$BT' && "
else
  INNER="cd '$BT' && "
fi

start_one() {
  local session="$1" py_script="$2" log_name="$3"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Skip: tmux session '$session' already exists (tmux attach -t $session)"
    return 0
  fi
  tmux new-session -d -s "$session" bash -lc "${INNER}python '$BT/$py_script' 2>&1 | tee '$BT/results/$log_name'"
  echo "Started: $session → results/$log_name"
}

start_one bell-tg  run_tg.py     run_tg.log
start_one bell-zz  run_ls_zz.py  run_ls_zz.log
start_one bell-xx  run_ls_xx.py  run_ls_xx.log

echo ""
echo "tmux ls:"
tmux ls 2>/dev/null | grep -E 'bell-(tg|zz|xx)' || true
echo "Attach: tmux attach -t bell-tg"
