# -*- coding: utf-8 -*-
"""不依赖 ultralytics exporter 的 TensorRT 构建与推理(TRT 10/11 tensor API)。"""
import numpy as np, torch, tensorrt as trt

LOGGER = trt.Logger(trt.Logger.WARNING)

_NP2T = {np.float32: torch.float32, np.float16: torch.float16,
         np.int32: torch.int32, np.int64: torch.int64,
         np.int8: torch.int8, np.uint8: torch.uint8, np.bool_: torch.bool}


def build_engine(onnx_path, out_path, fp16=False, int8=False, workspace_gb=8,
                 calibrator=None):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)          # TRT>=10: explicit batch 是唯一模式
    parser  = trt.OnnxParser(network, LOGGER)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print('  ONNX 解析错误:', parser.get_error(i))
            raise RuntimeError('ONNX 解析失败')
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
        if calibrator is not None:
            cfg.int8_calibrator = calibrator
    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        raise RuntimeError('引擎构建失败(build_serialized_network 返回 None)')
    with open(out_path, 'wb') as f:
        f.write(plan)
    return out_path


class TRTModel:
    """可直接当作 callable 用:out = model(x)。输入输出均为 cuda 上的 torch 张量。"""

    def __init__(self, engine_path, device='cuda:0'):
        with open(engine_path, 'rb') as f, trt.Runtime(LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f'反序列化失败: {engine_path}')
        self.ctx    = self.engine.create_execution_context()
        self.device = torch.device(device)
        self.inputs, self.outputs, self.buf = [], [], {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            mode  = self.engine.get_tensor_mode(n)
            shape = tuple(self.engine.get_tensor_shape(n))
            dt    = _NP2T[trt.nptype(self.engine.get_tensor_dtype(n))]
            self.buf[n] = torch.empty(shape, dtype=dt, device=self.device)
            (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs).append(n)
        for n in self.buf:
            self.ctx.set_tensor_address(n, self.buf[n].data_ptr())

        self.stream = torch.cuda.Stream(device=self.device)

    def __call__(self, x, *a, **k):
        n = self.inputs[0]
        with torch.cuda.stream(self.stream):
            self.buf[n].copy_(x.to(device=self.device, dtype=self.buf[n].dtype))
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        outs = [self.buf[n] for n in self.outputs]
        return outs[0] if len(outs) == 1 else outs
