#!/bin/bash
set -euo pipefail
D=/root/autodl-tmp/work/llm-infra-bench/days/day07
source "$D/scripts/env.sh"
printf '[1/3] Accepted layout and label validation\n'
/root/autodl-tmp/venvs/det/bin/python "$D/scripts/resume.py" 2>&1 | tee "$D/results/resume.log"
printf '[2/3] Task-local HTTP dependencies\n'
/root/autodl-tmp/venvs/det/bin/python -m pip install --target /root/autodl-tmp/sod/day07-deps -r "$D/scripts/requirements-http.txt" 2>&1 | tee "$D/results/deps.log"
printf '[3/3] Build on this GPU\n'
/root/autodl-tmp/venvs/det/bin/python "$D/scripts/build.py" 2>&1 | tee "$D/results/build.log"
