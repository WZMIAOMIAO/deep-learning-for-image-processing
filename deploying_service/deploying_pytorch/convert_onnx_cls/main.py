from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np
from model import resnet34

device = torch.device("cpu")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main(save_path=None):
    assert isinstance(save_path, str), "lack of save_path parameter..."
    # create model
    model = resnet34(num_classes=5)
    # load model weights
    model_weight_path = "./resNet34.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    # input to the model
    # [batch, channel, height, width]
    x = torch.rand(1, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    # export the model
    torch.onnx.export(model,                       # model being run
                      x,                           # model input (or a tuple for multiple inputs)
                      save_path,                   # where to save the model (can be a file or file-like object)
                      export_params=True,          # store the trained parameter weights inside the model file
                      opset_version=10,            # the ONNX version to export the model to
                      do_constant_folding=True,    # whether to execute constant folding for optimization
                      input_names=["input"],       # the model's input names
                      output_names=["output"],     # the model's output names
                      dynamic_axes={"input": {0: "batch_size"},  # variable length axes
                                    "output": {0: "batch_size"}})

    # check onnx model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(save_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and Pytorch results
    # assert_allclose: Raises an AssertionError if two objects are not equal up to desired tolerance.
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # load test image
    img = Image.open("../tulip.jpg")

    # pre-process
    preprocess = transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = preprocess(img)
    img = img.unsqueeze_(0)

    # feed image into onnx model
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    prediction = ort_outs[0]

    # np softmax process
    prediction -= np.max(prediction, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大元素
    prediction = np.exp(prediction) / np.sum(np.exp(prediction), keepdims=True)
    print(prediction)


if __name__ == '__main__':
    onnx_file_name = "resnet34.onnx"
    main(save_path=onnx_file_name)
