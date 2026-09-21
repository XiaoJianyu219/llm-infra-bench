#!/bin/bash
LOG=/root/autodl-tmp/work/serve_latest.log
echo "--- 参数是否生效 ---"
head -20 "$LOG" | grep -iE "^.*model |version"
grep -m1 "non-default args" "$LOG"
echo
echo "--- 显存分解（须与 baseline 一致）---"
grep -inE "loading took|Available KV cache|GPU KV cache size|Maximum concurrency" "$LOG"
echo
echo "--- 健康 ---"
curl -s -o /dev/null -w "health: %{http_code}\n" http://localhost:8000/health
