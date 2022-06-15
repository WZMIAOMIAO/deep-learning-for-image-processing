import cv2
import time
import numpy as np
import torch
from openvino.runtime import Core
from model import mobilenet_v3_large


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


def onnx_inference(onnx_path: str, image: np.ndarray, num_images: int = 20):
    # Load network to Inference Engine
    ie = Core()
    model_onnx = ie.read_model(model=onnx_path)
    compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

    input_layer_onnx = next(iter(compiled_model_onnx.inputs))
    output_layer_onnx = next(iter(compiled_model_onnx.outputs))

    start = time.perf_counter()
    request_onnx = compiled_model_onnx.create_infer_request()
    for _ in range(num_images):
        request_onnx.infer(inputs={input_layer_onnx.any_name: image})
    end = time.perf_counter()
    time_onnx = end - start
    print(
        f"ONNX model in Inference Engine/CPU: {time_onnx / num_images:.3f} "
        f"seconds per image, FPS: {num_images / time_onnx:.2f}"
    )


def ir_inference(ir_path: str, image: np.ndarray, num_images: int = 20):
    # Load the network in Inference Engine
    ie = Core()
    model_ir = ie.read_model(model=ir_path)
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

    # Get input and output layers
    input_layer_ir = next(iter(compiled_model_ir.inputs))
    output_layer_ir = next(iter(compiled_model_ir.outputs))

    start = time.perf_counter()
    request_ir = compiled_model_ir.create_infer_request()
    for _ in range(num_images):
        request_ir.infer(inputs={input_layer_ir.any_name: image})
    end = time.perf_counter()
    time_ir = end - start
    print(
        f"IR model in Inference Engine/CPU: {time_ir / num_images:.3f} "
        f"seconds per image, FPS: {num_images / time_ir:.2f}"
    )


def pytorch_inference(weights_path: str, image: np.ndarray, num_images: int = 20):
    image = torch.as_tensor(image, dtype=torch.float32)

    model = mobilenet_v3_large(num_classes=5)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(num_images):
            model(image)
        end = time.perf_counter()
        time_torch = end - start

    print(
        f"PyTorch model on CPU: {time_torch / num_images:.3f} seconds per image, "
        f"FPS: {num_images / time_torch:.2f}"
    )


def main():
    image_h = 224
    image_w = 224
    image_filename = "test.jpg"
    onnx_path = "mobilenet_v3.onnx"
    ir_path = "ir_output/mobilenet_v3.xml"
    pytorch_weights_path = "mbv3_flower.pth"

    image = cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(image, (image_w, image_h))
    normalized_image = normalize(resized_image)

    # Convert the resized images to network input shape
    # [h, w, c] -> [c, h, w] -> [1, c, h, w]
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
    normalized_input_image = np.expand_dims(np.transpose(normalized_image, (2, 0, 1)), 0)

    onnx_inference(onnx_path, normalized_input_image, num_images=50)
    ir_inference(ir_path, input_image, num_images=50)
    pytorch_inference(pytorch_weights_path, normalized_input_image, num_images=50)


if __name__ == '__main__':
    main()