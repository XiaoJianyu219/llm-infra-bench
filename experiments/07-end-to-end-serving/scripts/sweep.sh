#!/bin/bash
set -euo pipefail
D=/root/autodl-tmp/work/llm-infra-bench/days/day07
source "$D/scripts/env.sh"
/root/autodl-tmp/venvs/det/bin/python "$D/scripts/sweep.py" 2>&1 | tee "$D/results/sweep.log"
