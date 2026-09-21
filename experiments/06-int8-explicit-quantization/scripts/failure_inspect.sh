#!/bin/bash
DAY=/root/autodl-tmp/work/llm-infra-bench/days/day06
cp /root/autodl-tmp/sod/day06_failure_inspect.py "$DAY/scripts/failure_inspect.py"
cp /root/autodl-tmp/sod/day06_failure_inspect.sh "$DAY/scripts/failure_inspect.sh"
source "$DAY/scripts/env.sh"
/root/autodl-tmp/venvs/det/bin/python "$DAY/scripts/failure_inspect.py" > "$DAY/results/failure_inspect.log" 2>&1
cat "$DAY/results/failure_inspect.log"
