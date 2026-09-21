#!/bin/bash
set -euo pipefail
D=/root/autodl-tmp/work/llm-infra-bench/days/day07
source "$D/scripts/env.sh"
export ENGINE="${ENGINE:-/root/autodl-tmp/sod/day07_batch_fp16.engine}"
exec /root/autodl-tmp/venvs/det/bin/python -m uvicorn server:app --host 127.0.0.1 --port 18007 --no-access-log
