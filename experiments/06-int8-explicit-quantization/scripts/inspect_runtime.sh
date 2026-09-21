#!/bin/bash
set -u
DAY=/root/autodl-tmp/work/llm-infra-bench/days/day06
cp /root/autodl-tmp/sod/day06_inspect.py "$DAY/scripts/inspect_runtime.py"
cp /root/autodl-tmp/sod/day06_inspect.sh "$DAY/scripts/inspect_runtime.sh"
cp /root/autodl-tmp/sod/day06_preflight_retry.sh "$DAY/scripts/preflight_retry.sh"
export PYTHONDONTWRITEBYTECODE=1
export YOLO_CONFIG_DIR=/root/autodl-tmp/sod/day06-yolo-config
export MPLCONFIGDIR=/root/autodl-tmp/sod/day06-mpl
export XDG_CACHE_HOME=/root/autodl-tmp/sod/day06-cache
export TORCH_HOME=/root/autodl-tmp/sod/day06-torch
cd /root/autodl-tmp/sod || exit 1
git -C /root/autodl-tmp/work/llm-infra-bench status --short > "$DAY/results/git_before.txt"
command -v tmux > "$DAY/results/tmux_path.txt"
/root/autodl-tmp/venvs/det/bin/python "$DAY/scripts/inspect_runtime.py" > "$DAY/results/inspect.log" 2>&1
status=$?
cat "$DAY/results/inspect.log"
exit "$status"
