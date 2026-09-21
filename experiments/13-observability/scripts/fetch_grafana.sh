#!/bin/bash
# 重下 Grafana：dl.grafana.com 只有 ~6 KB/s，改走 GitHub release（network_turbo 加速的是 GitHub）
set -u
source /etc/network_turbo 2>/dev/null
B=/root/autodl-tmp/day13
cd $B
pkill -f 'curl.*grafana' 2>/dev/null
rm -f g.tgz
GV=13.2.2
for URL in \
  "https://github.com/grafana/grafana/releases/download/v$GV/grafana-$GV.linux-amd64.tar.gz" \
  "https://mirrors.tuna.tsinghua.edu.cn/grafana/oss/release/grafana-$GV.linux-amd64.tar.gz" \
  "https://mirrors.ustc.edu.cn/grafana/oss/release/grafana-$GV.linux-amd64.tar.gz"
do
  echo "试 $URL"
  timeout 600 curl -fL --no-progress-meter --connect-timeout 20 --speed-time 30 --speed-limit 200000 -o g.tgz "$URL" \
    && echo "OK 下到 $(du -h g.tgz | cut -f1)" && break
  echo "失败或太慢，换下一个"; rm -f g.tgz
done
if [ -s g.tgz ]; then
  tar xzf g.tgz && rm -f g.tgz
  ls -d grafana* 
  mv grafana-v$GV grafana 2>/dev/null || mv grafana-$GV grafana 2>/dev/null
  $B/grafana/bin/grafana --version 2>&1 | head -2
else
  echo "GRAFANA_DOWNLOAD_FAILED"
fi
echo GFETCH_DONE $(date "+%F %T")
