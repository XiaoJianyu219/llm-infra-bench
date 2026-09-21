import json,time
from pathlib import Path
D=Path('/root/autodl-tmp/work/llm-infra-bench/days/day07');R=D/'results'
def main():
 rows=json.loads((R/'nms_mismatches.json').read_text());out=[]
 for row in rows:
  if row['route']!='fp16_plugin_same_raw':continue
  a,b=row['ref'],row['actual'];sa={tuple(x) for x in a};sb={tuple(x) for x in b}
  lost=[list(x) for x in sa-sb];gained=[list(x) for x in sb-sa]
  item=dict(image=row['image'],ref_only=lost,plugin_only=gained,threshold_equal=[x for x in gained if x[4]==.25],same_score_pairs=[])
  for x in lost:
   for y in gained:
    if x[4:]==y[4:]:item['same_score_pairs'].append(dict(ref=x,plugin=y))
  out.append(item)
 (R/'plugin_diagnosis.json').write_text(json.dumps(out,indent=2))
 summary=dict(differing_images=len(out),threshold_equal_images=sum(bool(v['threshold_equal']) for v in out),tie_pair_images=sum(bool(v['same_score_pairs']) for v in out))
 with (D/'journal.md').open('a') as f:
  f.write('\n## '+time.strftime('%Y-%m-%d %H:%M:%S')+' 实测修正与插件诊断\n'
   'FP16 CPU链路预处理15.441ms、读图解码13.284ms，超过NMS2.279ms；原预测NMS占30–55%且最大被推翻。GPU NMS0.980ms，但端到端34.538→31.037ms。\n'
   '两档CPU/GPU检测结果各800/800图完全一致。FP16同raw插件788/800图完全一致，差异诊断：'+json.dumps(summary)+'。NVIDIA公开实现score阈值为小于时丢弃（等号保留），而本基线严格大于0.25；IoU边界以及并列分数顺序亦可能不同。公开main源码不是当前二进制的精确构建来源，机制解释须结合原框证据，不能仅凭源码推定。服务选择全量对齐的GPU batched_nms。\n'
   '动态batch1/4/8/16样本与同引擎单张raw最大差均为0；对PyTorch的80图检查高置信框最大偏差0.557输入像素、类别分数最大0.00667，4图框数不同，属于需保留的FP16/导出数值差异，未宣称服务与PyTorch无损。\n')
 print(json.dumps(summary));print(json.dumps(out[:2]))
if __name__=='__main__':main()
