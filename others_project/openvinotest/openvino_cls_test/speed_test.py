import os
import time
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import onnx
import onnxruntime
import numpy as np
from openvino.inference_engine import IECore
from model import resnet34

device = torch.device("cpu")


def check_path_exist(path):
    assert os.path.exists(path), "{} does not exist...".format(path)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def pytorch_model_speed(data_loader, val_num):
    net = resnet34(num_classes=5)
    # load weights
    model_weight_path = "./resNet34.pth"
    check_path_exist(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
    net.eval()
    test_data = torch.rand((1, 3, 224, 224))
    net(test_data.to(device))

    forward_time = 0
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in tqdm(data_loader, desc="Running pytorch model..."):
            val_images, val_labels = val_data
            t1 = time.time()
            outputs = net(val_images.to(device))  # eval model only have last output layer
            t2 = time.time()
            forward_time += (t2 - t1)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
    fps = round(val_num / forward_time, 1)
    print("pytorch info:\nfps: {}/s  accuracy: {}\n".format(fps,
                                                            val_accurate))
    return fps, val_accurate, "Pytorch(not opt)"


def onnx_model_speed(data_loader, val_num):
    # check onnx model
    onnx_path = "./resnet34.onnx"
    check_path_exist(onnx_path)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    forward_time = 0
    acc = 0.0  # accumulate accurate number / epoch
    for val_data in tqdm(data_loader, desc="Running onnx model..."):
        val_images, val_labels = val_data
        input_dict = {input_name: to_numpy(val_images)}
        t1 = time.time()
        outputs = ort_session.run(None, input_dict)
        t2 = time.time()
        forward_time += (t2 - t1)
        outputs = outputs[0]
        predict_y = np.argmax(outputs, axis=1)
        acc += (predict_y == to_numpy(val_labels)).sum()
    val_accurate = acc / val_num
    fps = round(val_num / forward_time, 1)
    print("onnx info:\nfps: {}/s  accuracy: {}\n".format(fps,
                                                         val_accurate))
    return fps, val_accurate, "ONNX"


def openvino_model_speed(data_loader, val_num):
    device = "CPU"
    model_xml_path = "./resnet34r.xml"
    model_bin_path = "./resnet34r.bin"
    check_path_exist(model_xml_path)
    check_path_exist(model_bin_path)

    # inference engine
    ie = IECore()

    # read IR
    net = ie.read_network(model=model_xml_path, weights=model_bin_path)
    # load model
    exec_net = ie.load_network(network=net, device_name=device)

    # check supported layers for device
    if device == "CPU":
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) > 0:
            print("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                  "or --cpu_extension command line argument")
            raise ValueError("device {} not support layers:\n {}".format(device,
                                                                         ",".join(not_supported_layers)))

    # get input and output name
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))

    # set batch size
    batch_size = 1
    net.batch_size = batch_size

    # read and pre-process input images
    # n, c, h, w = net.input_info[input_blob].input_data.shape
    forward_time = 0
    acc = 0.0  # accumulate accurate number / epoch
    for val_data in tqdm(data_loader, desc="Running onnx model..."):
        val_images, val_labels = val_data
        input_dict = {input_blob: to_numpy(val_images)}
        # start sync inference
        t1 = time.time()
        res = exec_net.infer(inputs=input_dict)
        t2 = time.time()
        forward_time += (t2 - t1)
        outputs = res[output_blob]
        predict_y = np.argmax(outputs, axis=1)
        acc += (predict_y == to_numpy(val_labels)).sum()
    val_accurate = acc / val_num
    fps = round(val_num / forward_time, 1)
    print("openvino info:\nfps: {}/s  accuracy: {}\n".format(fps,
                                                             val_accurate))


def main():
    data_transform = transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = "/home/w180662/my_project/my_github"  # get data root path
    image_path = os.path.join(data_root, "data_set/flower_data/")  # flower data set path
    check_path_exist(image_path)

    batch_size = 1

    validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                            transform=data_transform)
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    pytorch_model_speed(validate_loader, val_num)
    onnx_model_speed(validate_loader, val_num)
    openvino_model_speed(validate_loader, val_num)


if __name__ == '__main__':
    main()
