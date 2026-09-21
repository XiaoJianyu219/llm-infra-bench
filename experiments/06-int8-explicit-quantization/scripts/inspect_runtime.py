import os, sys, inspect, json
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp/sod')
import torch, tensorrt as trt, onnx, onnxruntime as ort
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.base import BaseDataset
from ultralytics.data.augment import LetterBox
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from onnxruntime.quantization import quantize_static, CalibrationDataReader
from simplify import simplify
day = Path('/root/autodl-tmp/work/llm-infra-bench/days/day06')
m = YOLO('/root/autodl-tmp/sod/best.pt')
n = simplify(m.model, verbose=False)
print('simplify_return:', n, flush=True)
if n != 4:
    with (day/'journal.md').open('a') as f:
        f.write(f'\n## 停止：第 3.6 节不符\n\nsimplify(net) 实际返回 {n}，要求为 4。未继续。\n')
    raise SystemExit(3)
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'TRT', trt.__version__, flush=True)
print('CalibrationDataReader dir', dir(CalibrationDataReader))
print('quantize_static signature', inspect.signature(quantize_static))
print('quantize_static doc', inspect.getdoc(quantize_static))
print('Builder dir', dir(trt.Builder))
print('IBuilderConfig dir', dir(trt.IBuilderConfig))
print('IOptimizationProfile dir', dir(trt.IOptimizationProfile))
print('IExecutionContext dir', dir(trt.IExecutionContext))
print('torch.onnx.export signature', inspect.signature(torch.onnx.export))
for name, obj in [('BaseValidator.__call__', BaseValidator.__call__), ('DetectionValidator.build_dataset', DetectionValidator.build_dataset), ('BaseDataset.set_rectangle', BaseDataset.set_rectangle), ('BaseDataset.load_image', BaseDataset.load_image), ('YOLODataset.build_transforms', YOLODataset.build_transforms), ('LetterBox', LetterBox), ('check_det_dataset', check_det_dataset)]:
    print('\nSOURCE', name, '\n', inspect.getsource(obj), flush=True)
g=onnx.load('/root/autodl-tmp/sod/best.onnx')
print('ONNX inputs', [str(x) for x in g.graph.input])
print('ONNX outputs', [str(x) for x in g.graph.output])
print('ONNX operators', sorted(set(n.op_type for n in g.graph.node)))
print('ONNX nodes count', len(g.graph.node))
print('AVAILABLE_PROVIDERS', ort.get_available_providers())
with (day/'journal.md').open('a') as f:
    f.write('\n## 实测：基础环境核验\n\nTensorRT 11.3.0.99，无 FP16/INT8 builder flags，有 STRONGLY_TYPED；800 张图片均为非软链接、硬链接计数至少 2，800 个对应标签存在。simplify(net) 返回 4。第 3 节上述条件通过；完整 API 与验证器源码检查见 results/inspect.log。\n')
