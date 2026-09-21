#!/bin/bash
set -u
DAY=/root/autodl-tmp/work/llm-infra-bench/days/day06
TAG=${1:-int8_full}
cp /root/autodl-tmp/sod/day06_int8_build.py "$DAY/scripts/int8_build.py"
cp /root/autodl-tmp/sod/day06_int8_build.sh "$DAY/scripts/int8_build.sh"
source "$DAY/scripts/env.sh"
/root/autodl-tmp/venvs/det/bin/python "$DAY/scripts/int8_build.py" "$TAG" > "$DAY/results/${TAG}_build.log" 2>&1
status=$?
printf '%s\n' "$status" > "$DAY/results/${TAG}_build.exit"
[ "$status" -eq 0 ] || exit "$status"
/root/autodl-tmp/venvs/det/bin/python "$DAY/scripts/eval.py" "$TAG" > "$DAY/results/${TAG}_eval.log" 2>&1
printf '%s\n' "$?" > "$DAY/results/${TAG}_eval.exit"
