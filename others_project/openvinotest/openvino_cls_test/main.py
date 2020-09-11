import sys
import cv2
import os
import glob
import json
import numpy as np
import logging as log
from openvino.inference_engine import IECore


def main():
    device = "CPU"
    model_xml_path = "./resnet34.xml"
    model_bin_path = "./resnet34.bin"
    image_path = "./"
    class_json_path = './class_indices.json'

    # set log format
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    assert os.path.exists(model_xml_path), ".xml file does not exist..."
    assert os.path.exists(model_bin_path), ".bin file does not exist..."

    # search *.jpg files
    image_list = glob.glob(os.path.join(image_path, "*.jpg"))
    assert len(image_list) > 0, "no image(.jpg) be found..."

    # load class label
    assert os.path.exists(class_json_path), "class_json_path does not exist..."
    json_file = open(class_json_path, 'r')
    class_indict = json.load(json_file)

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
            log.error("device {} not support layers:\n {}".format(device,
                                                                  ",".join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    # get input and output name
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))

    # set batch size
    batch_size = 1
    net.batch_size = batch_size

    # read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    # images = np.ndarray(shape=(n, c, h, w))
    # inference every image
    for i in range(len(image_list)):
        image = cv2.imread(image_list[i])
        if image.shape[:-1] != (h, w):
            image = cv2.resize(image, (w, h))
        # bgr(opencv default format) -> rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # pre-process
        image = (image / 255.).astype(np.float32)
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        # change data from HWC to CHW
        image = image.transpose((2, 0, 1))
        # add batch dimension
        image = np.expand_dims(image, axis=0)

        # start sync inference
        res = exec_net.infer(inputs={input_blob: image})
        prediction = np.squeeze(res[output_blob])
        # print(prediction)

        # np softmax process
        prediction -= np.max(prediction, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大元素
        prediction = np.exp(prediction) / np.sum(np.exp(prediction), keepdims=True)
        class_index = np.argmax(prediction, axis=0)
        print("prediction: '{}'\nclass:{}  probability:{}\n".format(image_list[i],
                                                                    class_indict[str(class_index)],
                                                                    np.around(prediction[class_index]), 2))


if __name__ == '__main__':
    main()
