import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from openvino.runtime import Core
from utils import letterbox, scale_coords, non_max_suppression, coco80_names
from draw_box_utils import draw_objs


def main():
    img_path = "test.jpg"
    ir_model_xml = "ir_output/yolov5s.xml"
    img_size = (640, 640)  # h, w

    origin_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    reshape_img, ratio, pad = letterbox(origin_img, img_size, auto=False)
    input_img = np.expand_dims(np.transpose(reshape_img, [2, 0, 1]), 0).astype(np.float32)

    # Load the network in Inference Engine
    ie = Core()
    model_ir = ie.read_model(model=ir_model_xml)
    compiled_model = ie.compile_model(model=model_ir, device_name="CPU")
    inputs_names = compiled_model.inputs
    outputs_names = compiled_model.outputs

    # inference
    request = compiled_model.create_infer_request()
    request.infer(inputs={inputs_names[0]: input_img})
    result = request.get_output_tensor(outputs_names[0].index).data

    # post-process
    result = non_max_suppression(torch.Tensor(result))[0]
    boxes = result[:, :4].numpy()
    scores = result[:, 4].numpy()
    cls = result[:, 5].numpy().astype(int)
    boxes = scale_coords(reshape_img.shape, boxes, origin_img.shape, (ratio, pad))

    draw_img = draw_objs(Image.fromarray(origin_img),
                         boxes,
                         cls,
                         scores,
                         category_index=dict([(str(i), v) for i, v in enumerate(coco80_names)]))
    plt.imshow(draw_img)
    plt.show()
    draw_img.save("predict.jpg")


if __name__ == '__main__':
    main()
