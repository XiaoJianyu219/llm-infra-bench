from common import *
from trt_utils import build_engine
import onnx, torch, tensorrt as trt, hashlib
from onnx import helper as h, TensorProto as T, numpy_helper as nh
from onnxconverter_common import float16
def plugin_graph(g,raw='output0'):
 # Match single-label argmax CPU policy before class-aware plugin NMS.
 g.graph.node.append(h.make_node('Transpose',[raw],['d7_pred'],perm=[0,2,1]))
 for name,val in [('d7_boxes_idx',[0,1,2,3]),('d7_scores_idx',[4,5,6,7]),('d7_class_idx',[0,1,2,3])]:
  g.graph.initializer.append(nh.from_array(np.array(val,dtype=np.int64),name))
 g.graph.initializer.append(nh.from_array(np.array(-1,dtype=np.float32),'d7_neg'))
 g.graph.node.extend([
  h.make_node('Gather',['d7_pred','d7_boxes_idx'],['d7_boxes'],axis=2),
  h.make_node('Gather',['d7_pred','d7_scores_idx'],['d7_scores'],axis=2),
  h.make_node('ArgMax',['d7_scores'],['d7_best'],axis=2,keepdims=1,select_last_index=0),
  h.make_node('Equal',['d7_best','d7_class_idx'],['d7_mask']),
  h.make_node('Where',['d7_mask','d7_scores','d7_neg'],['d7_single_scores']),
  h.make_node('EfficientNMS_TRT',['d7_boxes','d7_single_scores'],['num_dets','det_boxes','det_scores','det_classes'],domain='trt.plugins',plugin_version='1',plugin_namespace='',score_threshold=0.25,iou_threshold=0.7,max_output_boxes=300,background_class=-1,score_activation=0,class_agnostic=0,box_coding=1)])
 del g.graph.output[:]
 g.graph.output.extend([h.make_tensor_value_info('num_dets',T.INT32,[1,1]),h.make_tensor_value_info('det_boxes',T.FLOAT,[1,300,4]),h.make_tensor_value_info('det_scores',T.FLOAT,[1,300]),h.make_tensor_value_info('det_classes',T.INT32,[1,300])])
 g.opset_import.append(h.make_opsetid('trt.plugins',1))
 return g
def main():
 trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING),'')
 out={}
 for tag,src in [('fp16','best_fp16.onnx'),('int8','best_int8_qdq.onnx')]:
  print('BUILD',tag,flush=True)
  out[tag]=build_engine(S/src,S/f'day07_{tag}.engine');save('build.json',out)
 from ultralytics import YOLO
 from ultralytics.nn.modules.head import Detect
 from simplify import simplify
 m=YOLO(str(S/'best.pt')); n=simplify(m.model)
 if n!=4: journal('STOP 第5.6 simplify='+str(n));raise SystemExit(5)
 net=m.model.cuda().float().eval();net.fuse(verbose=False)
 for p in net.parameters():p.requires_grad_(False)
 for mod in net.modules():
  if isinstance(mod,Detect):mod.export=True;mod.format='onnx';mod.dynamic=True
 with torch.no_grad():
  torch.onnx.export(net,(torch.zeros(1,3,1024,1024,device='cuda'),),str(S/'day07_batch_fp32.onnx'),input_names=['images'],output_names=['output0'],opset_version=18,dynamo=False,dynamic_axes={'images':{0:'batch'},'output0':{0:'batch'}})
 del net,m
 torch.cuda.empty_cache()
 g=onnx.load(str(S/'day07_batch_fp32.onnx'))
 g=float16.convert_float_to_float16(g,keep_io_types=True,disable_shape_infer=False)
 onnx.save(g,str(S/'day07_batch_fp16.onnx'))
 out['batch']=build_engine(S/'day07_batch_fp16.onnx',S/'day07_batch_fp16.engine',dict(min=[1,3,1024,1024],opt=[4,3,1024,1024],max=[16,3,1024,1024]));save('build.json',out)
 try:
  g=plugin_graph(onnx.load(str(S/'best_fp16.onnx')))
  onnx.save(g,str(S/'day07_plugin.onnx'))
  out['plugin']=build_engine(S/'day07_plugin.onnx',S/'day07_plugin.engine')
  g=h.make_model(h.make_graph([], 'raw_nms',[h.make_tensor_value_info('output0',T.FLOAT,[1,8,87040])],[]),opset_imports=[h.make_opsetid('',18)])
  g.ir_version=10;g=plugin_graph(g)
  onnx.save(g,str(S/'day07_nms_only.onnx'))
  out['nms_only']=build_engine(S/'day07_nms_only.onnx',S/'day07_nms_only.engine')
 except Exception as e:
  out['plugin_failure']=str(e);journal('EfficientNMS 构建失败：'+str(e)+'；按4.2允许记录不可行原因。');traceback.print_exc()
 out['hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in S.glob('day07*.engine')}
 save('build.json',out); print('BUILD DONE',flush=True)
if __name__=='__main__':guarded(main)
