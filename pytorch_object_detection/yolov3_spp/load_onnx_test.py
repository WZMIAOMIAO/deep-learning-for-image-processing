import time
import cv2
import onnx
import onnxruntime
import numpy as np
from matplotlib import pyplot as plt
from draw_box_utils import draw_box


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def scale_img(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小，若需要填充，均匀填充到上下左右侧
    :param img: 输入的图像numpy格式
    :param new_shape: 输入网络的shape
    :param color: padding用什么颜色填充
    :param auto: 将输入网络的较小边长调整到最近的64整数倍(输入图像的比例不变)，这样输入网络的尺寸比指定尺寸要小，计算量也会减小
    :param scale_fill: 简单粗暴缩放到指定大小
    :param scale_up:  只缩小，不放大
    :return:
    """

    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍(416x416)，如果是(512x512)可以保证是64的整数倍
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def clip_coords(boxes: np.ndarray, img_shape: tuple):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2


def turn_back_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param coords: 预测的box信息
    :param img0_shape: 缩放前的图像尺度
    :param ratio_pad: 缩放过程中的缩放比例以及pad
    :return:
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def xywh2xyxy(x: np.ndarray):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def bboxes_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes: np.ndarray, iou_threshold=0.5, soft_threshold=0.3, sigma=0.5, method="nms", ) -> np.ndarray:
    """
    单独对一个类别进行NMS处理
    :param bboxes: [x1, y1, x2, y2, score]
    :param iou_threshold: nms算法中使用到的阈值
    :param soft_threshold: soft-nms算法中使用到的阈值
    :param sigma: soft-nms gaussian sigma
    :param method: nms或者soft-nms
    :return: 返回保留目标的索引
    """
    assert method in ["nms", "soft-nms"]
    # [x1, y1, x2, y2, score] -> [x1, y1, x2, y2, score, index]
    bboxes = np.concatenate([bboxes, np.arange(bboxes.shape[0]).reshape(-1, 1)], axis=1)

    best_bboxes_index = []
    while len(bboxes) > 0:
        max_ind = np.argmax(bboxes[:, 4])  # 寻找概率最大目标索引
        best_bbox = bboxes[max_ind]
        best_bboxes_index.append(best_bbox[5])
        bboxes = np.concatenate([bboxes[:max_ind], bboxes[max_ind + 1:]])  # 将最大概率目标去除
        ious = bboxes_iou(best_bbox[np.newaxis, :4], bboxes[:, :4])

        if method == "nms":
            iou_mask = np.less(ious, iou_threshold)  # <
        else:  # soft-nms
            weight = np.exp(-(np.square(ious) / sigma))
            bboxes[:, 4] = bboxes[:, 4] * weight
            iou_mask = np.greater(bboxes[:, 4], soft_threshold)  # >

        bboxes = bboxes[iou_mask]

    return np.array(best_bboxes_index, dtype=np.int8)


def post_process(pred: np.ndarray, multi_label=False, conf_thres=0.3):
    """
    输入的xywh都是归一化后的值
    :param pred: [num_obj, [x1, y1, x2, y2, objectness, cls1, cls1...]]
    :param img_size:
    :param multi_label:
    :param conf_thres:
    :return:
    """
    min_wh, max_wh = 2, 4096
    pred = pred[pred[:, 4] > conf_thres]  # 虑除小objectness目标
    pred = pred[((pred[:, 2:4] > min_wh) & (pred[:, 2:4] < max_wh)).all(1)]  # 虑除规定尺度范围外的目标

    if pred.shape[0] == 0:
        return np.empty((0, 6))  # [x, y, x, y, score, class]

    box = xywh2xyxy(pred[:, :4])
    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:  # 针对每个类别执行非极大值抑制
        # i, j = (x[:, 5:] > conf_thres).nonzero().t()
        # x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        pass
    else:  # best class only  直接针对每个类别中概率最大的类别进行非极大值抑制处理
        objectness = pred[:, 5:]
        class_index = np.argmax(objectness, axis=1)
        conf = objectness[(np.arange(pred.shape[0]), class_index)]
        # conf, j = predictions[:, 5:].max(1)
        pred = np.concatenate((box,
                               np.expand_dims(conf, axis=1),
                               np.expand_dims(class_index, axis=1)), 1)[conf > conf_thres]

    n = pred.shape[0]  # number of boxes
    if n == 0:
        return np.empty((0, 6))  # [x, y, x, y, score, class]

    cls = pred[:, 5]  # classes
    boxes, scores = pred[:, :4] + cls.reshape(-1, 1) * max_wh, pred[:, 4:5]
    t1 = time.time()
    indexes = nms(np.concatenate([boxes, scores], axis=1))
    print("NMS time is {}".format(time.time() - t1))
    pred = pred[indexes]

    return pred


def main():
    img_size = 512
    save_path = "yolov3spp.onnx"
    img_path = "test.jpg"
    input_size = (img_size, img_size)  # h, w

    # check onnx model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    # print(onnx.helper.printable_graph(onnx_model.graph))
    ort_session = onnxruntime.InferenceSession(save_path)

    img_o = cv2.imread(img_path)  # BGR
    assert img_o is not None, "Image Not Found " + img_path

    # preprocessing img
    img, ratio, pad = scale_img(img_o, new_shape=input_size, auto=False, color=(0, 0, 0))
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img).astype(np.float32)

    img /= 255.0  # scale (0, 255) to (0, 1)
    img = np.expand_dims(img, axis=0)  # add batch dimension

    # compute ONNX Runtime output prediction
    ort_inputs = {"images": img}

    t1 = time.time()
    # prediction: [num_obj, 85]
    pred = ort_session.run(None, ort_inputs)[0]
    t2 = time.time()
    print(t2 - t1)
    # print(predictions.shape[0])
    # process detections
    # 这里预测的数值是相对坐标(0-1之间)，乘上图像尺寸转回绝对坐标
    pred[:, [0, 2]] *= input_size[1]
    pred[:, [1, 3]] *= input_size[0]
    pred = post_process(pred)

    # 将预测的bbox缩放回原图像尺度
    p_boxes = turn_back_coords(img1_shape=img.shape[2:],
                               coords=pred[:, :4],
                               img0_shape=img_o.shape,
                               ratio_pad=[ratio, pad]).round()
    # print(p_boxes.shape)

    bboxes = p_boxes
    scores = pred[:, 4]
    classes = pred[:, 5].astype(np.int) + 1

    category_index = dict([(i + 1, str(i + 1)) for i in range(90)])
    img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
    plt.imshow(img_o)
    plt.show()


if __name__ == '__main__':
    main()
