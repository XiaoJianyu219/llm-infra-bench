# -*- coding: utf-8 -*-
"""TensorRT 11 下构建 FP16 引擎。

TRT 11 移除了弱类型网络,BuilderFlag 里不再有 FP16/INT8,
精度必须由 ONNX 图自身声明。故:
  1) 用 onnxconverter-common 把权重与算子转为 FP16(keep_io_types=True,
     输入输出仍是 fp32,对外接口不变)
  2) 用 NetworkDefinitionCreationFlag.STRONGLY_TYPED 构建
"""
import os, time, onnx, tensorrt as trt

SRC = '/root/autodl-tmp/sod/best.onnx'
F16 = '/root/autodl-tmp/sod/best_fp16.onnx'
OUT = '/root/autodl-tmp/sod/best_fp16.engine'
LOGGER = trt.Logger(trt.Logger.WARNING)

if not os.path.exists(F16):
    from onnxconverter_common import float16
    print('ONNX -> FP16 (keep_io_types=True) ...', flush=True)
    m16 = float16.convert_float_to_float16(onnx.load(SRC), keep_io_types=True)
    onnx.save(m16, F16)
print(f'FP16 ONNX: {os.path.getsize(F16)/2**20:.1f} MB '
      f'(FP32 ONNX {os.path.getsize(SRC)/2**20:.1f} MB)')

builder = trt.Builder(LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
parser  = trt.OnnxParser(network, LOGGER)
with open(F16, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print('  解析错误:', parser.get_error(i))
        raise SystemExit('ONNX 解析失败')

cfg = builder.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

t0 = time.time()
plan = builder.build_serialized_network(network, cfg)
if plan is None:
    raise SystemExit('引擎构建失败')
open(OUT, 'wb').write(plan)
print(f'FP16 引擎 {time.time()-t0:.1f}s -> {OUT}  {os.path.getsize(OUT)/2**20:.1f} MB')

# 快速自检:输入输出仍应是 fp32
import numpy as np
with open(OUT, 'rb') as f, trt.Runtime(LOGGER) as rt:
    eng = rt.deserialize_cuda_engine(f.read())
for i in range(eng.num_io_tensors):
    n = eng.get_tensor_name(i)
    print(f'  {n}: {eng.get_tensor_dtype(n)}  {tuple(eng.get_tensor_shape(n))}')
