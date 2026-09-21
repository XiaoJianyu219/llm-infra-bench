#!/bin/bash
# Run in tmux after preserving any previous run's results. This reruns measurements; it does not publish.
set -u
DAY=/root/autodl-tmp/work/llm-infra-bench/days/day06
source "$DAY/scripts/env.sh"
PY=/root/autodl-tmp/venvs/det/bin/python
"$PY" "$DAY/scripts/preflight.py" > "$DAY/results/reproduce_preflight.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/prepare.py" > "$DAY/results/reproduce_prepare.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/eval.py" baseline > "$DAY/results/reproduce_baseline.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/dynamic.py" > "$DAY/results/reproduce_dynamic_build.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/eval.py" dynamic > "$DAY/results/reproduce_dynamic_eval.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/quantize.py" > "$DAY/results/reproduce_int8_full_build.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/eval.py" int8_full > "$DAY/results/reproduce_int8_full_eval.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/int8_diagnose.py" > "$DAY/results/reproduce_int8_diagnose.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/exclude.py" > "$DAY/results/reproduce_exclude_plan.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/quantize.py" --exclude-head > "$DAY/results/reproduce_int8_head_build.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/eval.py" int8_head_excluded > "$DAY/results/reproduce_int8_head_eval.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/select_candidate.py" > "$DAY/results/reproduce_selection.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/measure.py" > "$DAY/results/reproduce_measure.log" 2>&1 || exit "$?"
"$PY" "$DAY/scripts/report.py" > "$DAY/results/reproduce_report.log" 2>&1 || exit "$?"
