import inspect
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.calibrate import MinMaxCalibrater
from ultralytics.nn.modules.head import Detect
import tensorrt as trt
for obj in [QDQQuantizer,MinMaxCalibrater,Detect,trt.OnnxParser]:
    print(obj.__name__,'dir',dir(obj),flush=True)
for obj in [QDQQuantizer.__init__,MinMaxCalibrater.create_inference_session,Detect.forward,Detect._inference]:
    print(inspect.getsource(obj),flush=True)
