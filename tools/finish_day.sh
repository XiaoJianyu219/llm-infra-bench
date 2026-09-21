#!/bin/bash
# 用法: ./finish_day.sh 06 "INT8 显式量化"
set -u
N=$(printf "%02d" "${1:?用法: ./finish_day.sh NN \"摘要\"}")
MSG="${2:?缺少一句话摘要}"
D="days/day$N"
cd "$(dirname "$0")" || exit 1

miss=0
for f in report.md journal.md; do
  [ -s "$D/$f" ] || { echo "缺少 $D/$f"; miss=1; }
done
[ -d "$D/results" ] && [ -n "$(ls -A $D/results 2>/dev/null)" ] || { echo "$D/results 为空"; miss=1; }
[ -d "$D/scripts" ] && [ -n "$(ls -A $D/scripts 2>/dev/null)" ] || { echo "$D/scripts 为空"; miss=1; }
[ "$miss" -ne 0 ] && { echo "四件套不完整, 未提交"; exit 1; }

echo "report=$(wc -c <$D/report.md)B  journal=$(wc -c <$D/journal.md)B  \
results=$(find $D/results -type f | wc -l)个  scripts=$(ls $D/scripts | wc -l)个"

source /etc/network_turbo 2>/dev/null
git add -A
git commit -m "Day $N: $MSG" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" || echo "无改动可提交"
git push && echo "已推送" || echo "推送失败, 稍后重试 git push"
