#!/bin/bash
# Run this from a VS Code Remote SSH terminal. No Codex/API is involved.
set -euo pipefail
D=/root/autodl-tmp/work/llm-infra-bench/days/day07
if tmux has-session -t day07scan 2>/dev/null || tmux has-session -t day07sweep 2>/dev/null; then
  echo 'A Day7 scan is already running. Use tmux attach -t day07scan, or inspect tmux list-sessions.'
  exit 1
fi
source "$D/scripts/env.sh"
/root/autodl-tmp/venvs/det/bin/python "$D/scripts/prepare_scan.py"
tmux new-session -d -s day07scan bash "$D/scripts/scan_and_report.sh"
echo 'Started tmux day07scan. Closing VS Code will not stop it.'
echo 'Progress: tail -n 5 /root/autodl-tmp/work/llm-infra-bench/days/day07/results/sweep.log'
echo 'Completion: cat /root/autodl-tmp/work/llm-infra-bench/days/day07/results/RUN_STATUS.json'
echo 'Expected completion: state=complete and validation.json passed=true.'
