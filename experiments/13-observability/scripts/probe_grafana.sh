#!/bin/bash
# 只探速度，不整包下载：各源限 12 秒，看能拿到多少字节
cd /root/autodl-tmp/day13
GV=13.2.2
probe(){
  local tag="$1" url="$2"
  local n
  n=$(timeout 12 curl -sS --no-progress-meter -r 0-20000000 -o /tmp/probe.bin -w '%{size_download}' "$url" 2>/tmp/probe.err)
  echo "$tag  拿到 ${n:-0} 字节/12s  $( [ -s /tmp/probe.err ] && head -c 120 /tmp/probe.err )"
}
echo "=== 不走 network_turbo ==="
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
probe "dl.grafana.com  " "https://dl.grafana.com/oss/release/grafana-$GV.linux-amd64.tar.gz"
echo "=== 走 network_turbo ==="
source /etc/network_turbo >/dev/null 2>&1
probe "dl.grafana.com  " "https://dl.grafana.com/oss/release/grafana-$GV.linux-amd64.tar.gz"
probe "github release  " "https://github.com/grafana/grafana/releases/download/v$GV/grafana-$GV.linux-amd64.tar.gz"
rm -f /tmp/probe.bin
