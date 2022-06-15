import cv2
import numpy as np
from openvino.runtime import Core


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
    # Load network to Inference Engine
    ie = Core()
    model_onnx = ie.read_model(model=onnx_path)
    compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

    input_layer_onnx = next(iter(compiled_model_onnx.inputs))
    output_layer_onnx = next(iter(compiled_model_onnx.outputs))

    # Run inference on the input image
    res_onnx = compiled_model_onnx(inputs=[image])[output_layer_onnx]
    return res_onnx


def ir_inference(ir_path: str, image: np.ndarray):
    # Load the network in Inference Engine
    ie = Core()
    model_ir = ie.read_model(model=ir_path)
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

    # Get input and output layers
    input_layer_ir = next(iter(compiled_model_ir.inputs))
    output_layer_ir = next(iter(compiled_model_ir.outputs))

    # Run inference on the input image
    res_ir = compiled_model_ir([image])[output_layer_ir]
    return res_ir


def main():
    image_h = 224
    image_w = 224
    image_filename = "test.jpg"
    onnx_path = "mobilenet_v3.onnx"
    ir_path = "ir_output/mobilenet_v3.xml"

    image = cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(image, (image_w, image_h))
    normalized_image = normalize(resized_image)

    # Convert the resized images to network input shape
    # [h, w, c] -> [c, h, w] -> [1, c, h, w]
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
    normalized_input_image = np.expand_dims(np.transpose(normalized_image, (2, 0, 1)), 0)

    onnx_res = onnx_inference(onnx_path, normalized_input_image)
    ir_res = ir_inference(ir_path, input_image)
    np.testing.assert_allclose(onnx_res, ir_res, rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with OpenvinoRuntime, and the result looks good!")


if __name__ == '__main__':
    main()
