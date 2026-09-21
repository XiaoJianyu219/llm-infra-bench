import inspect
import tensorrt as trt
from onnxruntime.quantization.calibrate import MinMaxCalibrater
print('set_shape doc:',trt.IOptimizationProfile.set_shape.__doc__)
print('get_shape doc:',trt.IOptimizationProfile.get_shape.__doc__)
print('set_input_shape doc:',trt.IExecutionContext.set_input_shape.__doc__)
print('set_tensor_address doc:',trt.IExecutionContext.set_tensor_address.__doc__)
print('add_optimization_profile doc:',trt.IBuilderConfig.add_optimization_profile.__doc__)
print(inspect.getsource(MinMaxCalibrater.collect_data))
print(inspect.getsource(MinMaxCalibrater.clear_collected_data))
print(inspect.getsource(MinMaxCalibrater.compute_data))
