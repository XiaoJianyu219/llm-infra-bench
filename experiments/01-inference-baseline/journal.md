# Day 01 日志

> **事后补记(2026-09-15)。** 当天没有单独写日志,本文件由对话记录与
> `report.md` 回溯整理。为遵守 CONVENTION 第 4 条(预测先于测量),
> 下列"当时的判断"只记录确实在测量前作出、且事后被数据推翻的判断,
> 不补写任何未曾说出口的预测。

## 当时判断错的地方

**(a) 以为显存峰值会随并发上升。** 实测不会 —— vLLM 启动时就把 KV cache
池一次性预分配好了,并发变化只改变池内已分配块的占用率,`nvidia-smi` 看到的
峰值几乎不动。教训:看 allocator 的分配策略,不要用"请求变多所以显存变多"
这种直觉外推。

## 环境与流程上踩的坑

- `uv` 未安装;`pip install uv` 又因为 AutoDL 学术加速把 http 的 aliyun 镜像
  代理掉而失败。解法:`unset http_proxy https_proxy all_proxy` 后改用
  https 的清华源。
- uv 不读 pip 的镜像配置,需单独设 `UV_DEFAULT_INDEX` / `UV_INDEX_URL`。
- `max-model-len=16384` 起服务失败:KV cache 需 2.25 GiB,可用仅 1.96 GiB。
  降到 8192 并把 `gpu-memory-utilization` 提到 0.94 后正常。
- `vllm bench serve` 对 hf-mirror 返回 401 —— `--model` 同时用于请求体和
  本地加载 tokenizer/config。去掉 `--served-model-name`,两处统一用绝对路径。
- **服务实际加载的是默认模型而非指定模型**,因为 `$MODEL` 在那个早于
  `.bashrc` 修改的窗口里是空的。教训:每次起服务后必须核对启动日志里的
  `model` 字段和 `non-default args`;变量一律在脚本内部定义并加 `set -eu`。
- `sweep.sh` 在 warmup 后静默退出:`set -e` 叠加 `>/dev/null 2>&1` 把
  错误吞掉了(venv 未激活)。改为脚本自行激活 venv、起跑前做健康检查、
  失败时打印 warmup 日志尾部。

## 当天确立并沿用至今的测量纪律

固定随机种子、固定 num-prompts、丢弃 warmup、每点重复 3 次取**中位数**
(不取最好值)、后台采样显存、每天快照环境、按区间分别估噪声地板。

