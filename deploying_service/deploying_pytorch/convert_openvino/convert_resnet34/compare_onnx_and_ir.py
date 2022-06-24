import numpy as np
import onnxruntime
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
    # load onnx model
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # compute onnx Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    res_onnx = ort_session.run(None, ort_inputs)[0]
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
    onnx_path = "resnet34.onnx"
    ir_path = "ir_output/resnet34.xml"

    image = np.random.randn(image_h, image_w, 3)
    normalized_image = normalize(image)

    # Convert the resized images to network input shape
    # [h, w, c] -> [c, h, w] -> [1, c, h, w]
    input_image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
    normalized_input_image = np.expand_dims(np.transpose(normalized_image, (2, 0, 1)), 0)

    onnx_res = onnx_inference(onnx_path, normalized_input_image)
    ir_res = ir_inference(ir_path, input_image)
    np.testing.assert_allclose(onnx_res, ir_res, rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with OpenvinoRuntime, and the result looks good!")


if __name__ == '__main__':
    main()
