"""Day 5 tensor API wrapper extended for explicit QDQ and dynamic input shapes."""
import time
from pathlib import Path
import numpy as np
import torch
import tensorrt as trt
LOGGER=trt.Logger(trt.Logger.WARNING)
_DT={np.float32:torch.float32,np.float16:torch.float16,np.int32:torch.int32,np.int64:torch.int64,np.int8:torch.int8,np.uint8:torch.uint8,np.bool_:torch.bool}

def build_engine(onnx_path,out_path,profile=None,workspace_gb=8):
    assert hasattr(trt.NetworkDefinitionCreationFlag,'STRONGLY_TYPED')
    assert not hasattr(trt.BuilderFlag,'FP16') and not hasattr(trt.BuilderFlag,'INT8')
    builder=trt.Builder(LOGGER)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser=trt.OnnxParser(network,LOGGER)
    if not parser.parse(Path(onnx_path).read_bytes()):
        for i in range(parser.num_errors): print(parser.get_error(i),flush=True)
        raise RuntimeError('ONNX parsing failed')
    cfg=builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,workspace_gb<<30)
    if profile is not None:
        p=builder.create_optimization_profile()
        p.set_shape(network.get_input(0).name,profile['min'],profile['opt'],profile['max'])
        assert [list(s) for s in p.get_shape(network.get_input(0).name)]==[profile[k] for k in ['min','opt','max']]
        assert cfg.add_optimization_profile(p)>=0
    t=time.perf_counter()
    plan=builder.build_serialized_network(network,cfg)
    if plan is None: raise RuntimeError('TensorRT engine build returned None')
    Path(out_path).write_bytes(bytes(plan))
    return {'seconds':time.perf_counter()-t,'bytes':Path(out_path).stat().st_size,'profile':profile,'strongly_typed':True}

class TRTModel:
    def __init__(self,path):
        with trt.Runtime(LOGGER) as rt: self.engine=rt.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None: raise RuntimeError('Engine deserialization failed')
        self.ctx=self.engine.create_execution_context()
        self.inputs=[]; self.outputs=[]; self.buf={}
        for i in range(self.engine.num_io_tensors):
            n=self.engine.get_tensor_name(i)
            (self.inputs if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT else self.outputs).append(n)
        assert len(self.inputs)==1
        self.stream=torch.cuda.Stream()
        self.last_shape=None
    def __call__(self,x,*args,**kwargs):
        shape=tuple(x.shape)
        if shape!=self.last_shape:
            static=tuple(self.engine.get_tensor_shape(self.inputs[0]))
            if -1 not in static and shape!=static: raise ValueError(f'Input {shape} != engine {static}')
            if -1 in static and not self.ctx.set_input_shape(self.inputs[0],shape): raise ValueError(f'Shape rejected {shape}')
            for n in self.inputs+self.outputs:
                s=tuple(self.ctx.get_tensor_shape(n)); assert all(v>0 for v in s), (n,s)
                self.buf[n]=torch.empty(s,dtype=_DT[trt.nptype(self.engine.get_tensor_dtype(n))],device='cuda:0')
                assert self.ctx.set_tensor_address(n,self.buf[n].data_ptr())
            self.last_shape=shape
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.buf[self.inputs[0]].copy_(x)
            if not self.ctx.execute_async_v3(self.stream.cuda_stream): raise RuntimeError('TensorRT execution failed')
        self.stream.synchronize()
        out=[self.buf[n] for n in self.outputs]
        return out[0] if len(out)==1 else out
