#!/bin/bash
set -u
B=/root/autodl-tmp/day13
cd $B
IMG=$(ls /root/autodl-tmp/sod/dataset800/images/*.png /root/autodl-tmp/sod/dataset800/images/*/*.png 2>/dev/null | head -1)
[ -z "$IMG" ] && IMG=$(find /root/autodl-tmp/sod/dataset800 -name '*.png' -o -name '*.jpg' | head -1)
echo "测试图: $IMG  ($(du -h "$IMG" 2>/dev/null | cut -f1))"
date "+加压开始 %F %T"
/root/autodl-tmp/venvs/det/bin/python loadgen13.py \
  --url http://127.0.0.1:18013/predict --img "$IMG" \
  --stages 1:60,2:60,4:60,8:60,16:60 --idle-tail 45 \
  --out results/loadgen.jsonl 2>&1 | grep -v '^   t=' | tail -20
date "+加压结束 %F %T"
echo LOAD13_DONE
