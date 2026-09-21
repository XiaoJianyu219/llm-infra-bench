#!/bin/bash
set -u
DAY=/root/autodl-tmp/work/llm-infra-bench/days/day06
cp /root/autodl-tmp/sod/day06_quantize.py "$DAY/scripts/quantize.py"
cp /root/autodl-tmp/sod/day06_quantize.sh "$DAY/scripts/quantize.sh"
source "$DAY/scripts/env.sh"
/root/autodl-tmp/venvs/det/bin/python "$DAY/scripts/quantize.py" --no-build >> "$DAY/results/int8_full_quantize.log" 2>&1
printf '%s\n' "$?" > "$DAY/results/int8_full_quantize.exit"
