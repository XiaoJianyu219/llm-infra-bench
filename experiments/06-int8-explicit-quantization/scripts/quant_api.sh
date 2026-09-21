#!/bin/bash
DAY=/root/autodl-tmp/work/llm-infra-bench/days/day06
cp /root/autodl-tmp/sod/day06_quant_api.py "$DAY/scripts/quant_api.py"
cp /root/autodl-tmp/sod/day06_quant_api.sh "$DAY/scripts/quant_api.sh"
source "$DAY/scripts/env.sh"
/root/autodl-tmp/venvs/det/bin/python "$DAY/scripts/quant_api.py" > "$DAY/results/quant_api.log" 2>&1
cat "$DAY/results/quant_api.log"
