from common import *
import shutil,hashlib
def main():
    tag='int8_head_excluded'
    d=json.loads((R/f'metrics_{tag}.json').read_text())
    selected={'tag':tag,'engine':str(SOD/f'{tag}.engine'),'onnx':str(SOD/f'{tag}.onnx'),'map':d['map'],'status':'experimental_candidate_pending_alignment','selection_reason':'Prespecified head-exclusion diagnostic after the full-QDQ output probabilities collapsed to zero; no further eval-set tuning.'}
    save('selected_int8.json',selected)
    for ext in ['onnx','engine']: shutil.copy2(SOD/f'{tag}.{ext}',SOD/f'best_int8_qdq.{ext}')
    data={}
    for name in ['best_int8_qdq.onnx','best_int8_qdq.engine','int8_full.onnx','int8_full.engine','int8_head_excluded.onnx','int8_head_excluded.engine','day06_dynamic.onnx','day06_dynamic.engine','day06_fixed1056.onnx','day06_fixed1056.engine']:
        p=SOD/name; data[name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    save('deliverable_artifacts.json',data)
    journal('## 后续测量候选固定\n'+json.dumps(selected,ensure_ascii=False)+'\n保存命名产物不代表零翻转验收通过。接下来的80张真实图对齐、背景图、单会话延迟沿用已记录预测；全量失败INT8也纳入诊断对照。')
    print('SELECTED',json.dumps(selected),flush=True)
if __name__=='__main__': guarded(main)
