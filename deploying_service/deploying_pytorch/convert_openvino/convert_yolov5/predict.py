import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from openvino.runtime import Core
from openvino.runtime.opset8 import multiclass_nms
from utils import letterbox, scale_coords, non_max_suppression, xywh2xyxy, coco80_names
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

    # boxes = result[0, :, :4]
    # # Compute conf
    # result[:, :, 5:] *= result[:, :, 4:5]  # conf = obj_conf * cls_conf
    # # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    # boxes = xywh2xyxy(boxes)
    # scores = result[:, :, 5:]
    # s = multiclass_nms(boxes=boxes[None], scores=scores, iou_threshold=0.45, score_threshold=0.25, keep_top_k=100)

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
