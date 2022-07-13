import numpy as np
import tensorrt as trt
import onnxruntime
import pycuda.driver as cuda
import pycuda.autoinit


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image to the given mean and standard deviation
    """
    image = image.astype(np.float32)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image /= 255.0
    image -= mean
    image /= std
    return image


def onnx_inference(onnx_path: str, image: np.ndarray):
    # load onnx model
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # compute onnx Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    res_onnx = ort_session.run(None, ort_inputs)[0]
    return res_onnx


def trt_inference(trt_path: str, image: np.ndarray):
    # Load the network in Inference Engine
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(trt_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        # Set input shape based on image dimensions for inference
        context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image.shape[-2], image.shape[-1]))
        # Allocate host and device buffers
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(image)
                input_memory = cuda.mem_alloc(image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()

        res_trt = np.reshape(output_buffer, (1, -1))

    return res_trt


def main():
    image_h = 224
    image_w = 224
    onnx_path = "resnet34.onnx"
    trt_path = "trt_output/resnet34.trt"

    image = np.random.randn(image_h, image_w, 3)
    normalized_image = normalize(image)

    # Convert the resized images to network input shape
    # [h, w, c] -> [c, h, w] -> [1, c, h, w]
    normalized_image = np.expand_dims(np.transpose(normalized_image, (2, 0, 1)), 0)

    onnx_res = onnx_inference(onnx_path, normalized_image)
    ir_res = trt_inference(trt_path, normalized_image)
    np.testing.assert_allclose(onnx_res, ir_res, rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with TensorRT Runtime, and the result looks good!")


if __name__ == '__main__':
    main()
