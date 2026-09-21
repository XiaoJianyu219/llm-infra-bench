#!/bin/bash
# Day13 3.1：拉 Prometheus 与 Grafana 的单二进制（AutoDL 容器内起不了 dockerd，见 Day12）
set -u
source /etc/network_turbo 2>/dev/null
B=/root/autodl-tmp/day13
mkdir -p $B && cd $B
step(){ echo; echo "########## $* ##########"; date "+%H:%M:%S"; }

step "1. 查真实版本号（不猜版本，Day12 踩过）"
PV=$(curl -fsSL --max-time 30 https://api.github.com/repos/prometheus/prometheus/releases/latest | grep -m1 '"tag_name"' | sed 's/.*"v\([0-9.]*\)".*/\1/')
echo "prometheus 最新版: ${PV:-查询失败}"
PV=${PV:-2.53.2}

step "2. 下 Prometheus"
if [ ! -x $B/prometheus/prometheus ]; then
  curl -fL --max-time 600 -o p.tgz \
    https://github.com/prometheus/prometheus/releases/download/v$PV/prometheus-$PV.linux-amd64.tar.gz \
    && tar xzf p.tgz && mv prometheus-$PV.linux-amd64 prometheus && rm -f p.tgz
fi
$B/prometheus/prometheus --version 2>&1 | head -2

step "3. 下 Grafana"
GV=$(curl -fsSL --max-time 30 https://api.github.com/repos/grafana/grafana/releases/latest | grep -m1 '"tag_name"' | sed 's/.*"v\([0-9.]*\)".*/\1/')
echo "grafana 最新版: ${GV:-查询失败}"
GV=${GV:-11.2.0}
if [ ! -x $B/grafana/bin/grafana ]; then
  curl -fL --max-time 900 -o g.tgz https://dl.grafana.com/oss/release/grafana-$GV.linux-amd64.tar.gz \
    && tar xzf g.tgz && mv grafana-v$GV grafana 2>/dev/null || mv grafana-$GV grafana 2>/dev/null
  rm -f g.tgz
fi
ls -d $B/grafana 2>/dev/null && $B/grafana/bin/grafana --version 2>&1 | head -2
du -sh $B/* 2>/dev/null
echo FETCH_DONE $(date "+%F %T")
