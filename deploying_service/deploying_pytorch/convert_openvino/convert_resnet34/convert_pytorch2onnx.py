import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np
from torchvision.models import resnet34

device = torch.device("cpu")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def main():
    weights_path = "resNet34(flower).pth"
    onnx_file_name = "resnet34.onnx"
    batch_size = 1
    img_h = 224
    img_w = 224
    img_channel = 3

    # create model and load pretrain weights
    model = resnet34(pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    model.eval()
    # input to the model
    # [batch, channel, height, width]
    x = torch.rand(batch_size, img_channel, img_h, img_w, requires_grad=True)
    torch_out = model(x)

    # export the model
    torch.onnx.export(model,             # model being run
                      x,                 # model input (or a tuple for multiple inputs)
                      onnx_file_name,    # where to save the model (can be a file or file-like object)
                      verbose=False)

    # check onnx model
    onnx_model = onnx.load(onnx_file_name)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_file_name)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and Pytorch results
    # assert_allclose: Raises an AssertionError if two objects are not equal up to desired tolerance.
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == '__main__':
    main()
