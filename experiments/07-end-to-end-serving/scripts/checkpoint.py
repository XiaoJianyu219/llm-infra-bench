import json,time,shutil,hashlib
from pathlib import Path
D=Path('/root/autodl-tmp/work/llm-infra-bench/days/day07');R=D/'results';S=Path('/root/autodl-tmp/sod')
def main():
 done=sorted(R.glob('b*_w*_c*_r*.json'))
 (R/'checkpoint.json').write_text(json.dumps(dict(updated=time.strftime('%Y-%m-%d %H:%M:%S'),status='running_sweep',completed_runs=len(done),total_runs=192,measure_complete=True,run_files=[p.name for p in done],resume='If tmux day07sweep is alive, observe it; do not launch duplicate. Do not combine independent sessions for speed ratios.'),indent=2))
 if not (R/'sweep.json').exists():
  (D/'report.md').write_text('# Day 07 — 实验进行中\n\n用户已接受标签布局并授权继续，先前停止已解除。\n\n已完成：七段配对计时、800图NMS对齐、新GPU引擎重建和动态batch样本验证。FP16 CPU链路端到端34.538ms，GPU后处理31.037ms；CPU/GPU NMS两档均800/800图完全一致。FP16插件同raw仅788/800图完全一致，差异全部保留。\n\nHTTP完整扫描进行中：'+str(len(done))+'/192次；最终报告将在全部测量结束后生成。进度见results/checkpoint.json，原始实测见results/timing.json与nms_alignment.json。尚未提交推送，未声称Day7完成。\n')
 print(len(done))
if __name__=='__main__':main()
