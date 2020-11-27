import os
import torch
import cv2
import torch.onnx
import onnx
import onnxruntime
import numpy as np
import models
from utils import img_utils

device = torch.device("cpu")
models.ONNX_EXPORT = True


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/yolov3-spp.cfg"
    weights = "weights/yolov3-spp-ultralytics-{}.pt".format(img_size)
    assert os.path.exists(cfg), "cfg file does not exist..."
    assert os.path.exists(weights), "weights file does not exist..."

    input_size = (img_size, img_size)  # [h, w]

    # create model
    model = models.Darknet(cfg, input_size)
    # load model weights
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.to(device)
    model.eval()
    # input to the model
    # [batch, channel, height, width]
    # x = torch.rand(1, 3, *input_size, requires_grad=True)
    img_path = "test.jpg"
    img_o = cv2.imread(img_path)  # BGR
    assert img_o is not None, "Image Not Found " + img_path

    # preprocessing img
    img = img_utils.letterbox(img_o, new_shape=input_size, auto=False, color=(0, 0, 0))[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img).astype(np.float32)

    img /= 255.0  # scale (0, 255) to (0, 1)
    img = np.expand_dims(img, axis=0)  # add batch dimension
    x = torch.tensor(img)
    torch_out = model(x)

    save_path = "yolov3spp.onnx"
    # export the model
    torch.onnx.export(model,                       # model being run
                      x,                           # model input (or a tuple for multiple inputs)
                      save_path,                   # where to save the model (can be a file or file-like object)
                      export_params=True,          # store the trained parameter weights inside the model file
                      opset_version=12,            # the ONNX version to export the model to
                      do_constant_folding=True,    # whether to execute constant folding for optimization
                      input_names=["images"],       # the model's input names
                      # output_names=["classes", "boxes"],     # the model's output names
                      output_names=["prediction"],
                      dynamic_axes={"images": {0: "batch_size"},  # variable length axes
                                    "prediction": {0: "batch_size"}})
                                    # "classes": {0: "batch_size"},
                                    # "confidence": {0: "batch_size"},
                                    # "boxes": {0: "batch_size"}})

    # check onnx model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))

    ort_session = onnxruntime.InferenceSession(save_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {"images": to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and Pytorch results
    # assert_allclose: Raises an AssertionError if two objects are not equal up to desired tolerance.
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)
    # np.testing.assert_allclose(to_numpy(torch_out[2]), ort_outs[2], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    main()
