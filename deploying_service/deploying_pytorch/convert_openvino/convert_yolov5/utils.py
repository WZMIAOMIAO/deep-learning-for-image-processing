import os
import time
import json
import copy

import cv2
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from compression.api import DataLoader, Metric


coco80_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


class MAPMetric(Metric):
    def __init__(self, map_value="map", conf_thres=0.001, iou_thres=0.6):
        """
        Mean Average Precision Metric. Wraps torchmetrics implementation, see
        https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html

        :map_value: specific metric to return. Default: "map"
                    Change `to one of the values in the list below to return a different value
                    ['mar_1', 'mar_10', 'mar_100', 'mar_small', 'mar_medium', 'mar_large',
                     'map', 'map_50', 'map_75', 'map_small', 'map_medium', 'map_large']
                    See torchmetrics documentation for more details.
        """

        self._name = map_value
        self.metric = MeanAveragePrecision(box_format="xyxy")
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        super().__init__()

    @property
    def value(self):
        """
        Returns metric value for the last model output.
        Possible format: {metric_name: [metric_values_per_image]}
        """
        return {self._name: [0]}

    @property
    def avg_value(self):
        """
        Returns average metric value for all model outputs.
        Possible format: {metric_name: metric_value}
        """
        return {self._name: self.metric.compute()[self._name].item()}

    def update(self, output, target):
        """
        Convert network output and labels to the format that torchmetrics' MAP
        implementation expects, and call `metric.update()`.

        :param output: model output
        :param target: annotations for model output
        """
        targetboxes = []
        targetlabels = []
        predboxes = []
        predlabels = []
        scores = []

        for single_target in target[0]:
            txmin, tymin, txmax, tymax = single_target["bbox"]
            category = single_target["category_id"]

            targetbox = [round(txmin), round(tymin), round(txmax), round(tymax)]
            targetboxes.append(targetbox)
            targetlabels.append(category)

        output = torch.Tensor(output[0]).float()
        output = non_max_suppression(output, conf_thres=self.conf_thres, iou_thres=self.iou_thres, multi_label=True)
        for single_output in output:
            for pred in single_output.numpy():
                xmin, ymin, xmax, ymax, conf, label = pred

                predbox = [round(xmin), round(ymin), round(xmax), round(ymax)]
                predboxes.append(predbox)
                predlabels.append(label)
                scores.append(conf)

        preds = [
            dict(
                boxes=torch.Tensor(predboxes).float(),
                labels=torch.Tensor(predlabels).short(),
                scores=torch.Tensor(scores),
            )
        ]
        targets = [
            dict(
                boxes=torch.Tensor(targetboxes).float(),
                labels=torch.Tensor(targetlabels).short(),
            )
        ]
        self.metric.update(preds, targets)

    def reset(self):
        """
        Resets metric
        """
        self.metric.reset()

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {"direction": "higher-better", "type": "mAP"}}


def _coco_remove_images_without_annotations(dataset, ids):
    """
    删除coco数据集中没有目标，或者目标面积非常小的数据
    refer to:
    https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
    :param dataset:
    :param cat_list:
    :return:
    """
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False

        return True

    valid_ids = []
    for ds_idx, img_id in enumerate(ids):
        ann_ids = dataset.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.loadAnns(ann_ids)

        if _has_valid_annotation(anno):
            valid_ids.append(img_id)

    return valid_ids


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        assert ratio_pad[0][0] == ratio_pad[0][1]
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (left, top)


class MyDataLoader(DataLoader):
    """`MS Coco Detection <https://cocodataset.org/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        dataset (string): "train" or "val.
        size (tuple): (h, w)
    """
    def __init__(self, root, dataset="train", size=(640, 640)):
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        anno_file = "instances_{}2017.json".format(dataset)
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root, "{}2017".format(dataset))
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.mode = dataset
        self.size = size
        self.coco = COCO(self.anno_path)

        self.coco91_id2classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])
        coco90_classes2id = dict([(v["name"], v["id"]) for k, v in self.coco.cats.items()])

        self.coco80_classes = coco80_names
        self.coco_id80_to_id91 = dict([(i, coco90_classes2id[k]) for i, k in enumerate(coco80_names)])

        ids = list(sorted(self.coco.imgs.keys()))

        # 移除没有目标，或者目标面积非常小的数据
        valid_ids = _coco_remove_images_without_annotations(self.coco, ids)
        self.ids = valid_ids

    def parse_targets(self,
                      coco_targets: list,
                      w: int = None,
                      h: int = None,
                      ratio: tuple = None,
                      pad: tuple = None):
        assert w > 0
        assert h > 0

        # 只筛选出单个对象的情况
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], a_min=0, a_max=w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], a_min=0, a_max=h)

        classes = [self.coco80_classes.index(self.coco91_id2classes[obj["category_id"]])
                   for obj in anno]
        classes = np.array(classes, dtype=int)

        # 筛选出合法的目标，即x_max>x_min且y_max>y_min
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        if ratio is not None:
            # width, height ratios
            boxes[:, 0::2] *= ratio[0]
            boxes[:, 1::2] *= ratio[1]

        if pad is not None:
            # dw, dh padding
            dw, dh = pad
            boxes[:, 0::2] += dw
            boxes[:, 1::2] += dh

        target_annotations = []
        for i in range(boxes.shape[0]):
            target_annotation = {
                "category_id": int(classes[i]),
                "bbox": boxes[i].tolist()
            }
            target_annotations.append(target_annotation)

        return target_annotations

    def __getitem__(self, index):
        """
        Get an item from the dataset at the specified index.
        Detection boxes are converted from absolute coordinates to relative coordinates
        between 0 and 1 by dividing xmin, xmax by image width and ymin, ymax by image height.

        :return: (annotation, input_image, metadata) where annotation is (index, target_annotation)
                 with target_annotation as a dictionary with keys category_id, image_width, image_height
                 and bbox, containing the relative bounding box coordinates [xmin, ymin, xmax, ymax]
                 (with values between 0 and 1) and metadata a dictionary: {"filename": path_to_image}
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        image_path = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.img_root, image_path))

        origin_h, origin_w, c = img.shape
        image, ratio, pad = letterbox(img, auto=False, new_shape=self.size)
        target_annotations = self.parse_targets(coco_target, origin_w, origin_h, ratio, pad)

        item_annotation = (index, target_annotations)
        input_image = np.expand_dims(image.transpose(2, 0, 1), axis=0).astype(
            np.float32
        )
        return (
            item_annotation,
            input_image,
            {"filename": str(image_path),
             "origin_shape": img.shape,
             "shape": image.shape,
             "img_id": img_id,
             "ratio_pad": [ratio, pad]},
        )

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collate_fn(x):
        return x


class EvalCOCOMetric:
    def __init__(self,
                 coco: COCO = None,
                 iou_type: str = "bbox",
                 results_file_name: str = "predict_results.json",
                 classes_mapping: dict = None):
        self.coco = copy.deepcopy(coco)
        self.results = []
        self.classes_mapping = classes_mapping
        self.coco_evaluator = None
        assert iou_type in ["bbox"]
        self.iou_type = iou_type
        self.results_file_name = results_file_name

    def prepare_for_coco_detection(self, ann, output):
        """将预测的结果转换成COCOeval指定的格式，针对目标检测任务"""
        # 遍历每张图像的预测结果
        if len(output[0]) == 0:
            return

        img_id = ann[2]["img_id"]
        per_image_boxes = output[0]
        per_image_boxes = scale_coords(img1_shape=ann[2]["shape"],
                                       coords=per_image_boxes,
                                       img0_shape=ann[2]["origin_shape"],
                                       ratio_pad=ann[2]["ratio_pad"])
        # 对于coco_eval, 需要的每个box的数据格式为[x_min, y_min, w, h]
        # 而我们预测的box格式是[x_min, y_min, x_max, y_max]，所以需要转下格式
        per_image_boxes[:, 2:] -= per_image_boxes[:, :2]
        per_image_classes = output[1].tolist()
        per_image_scores = output[2].tolist()

        # 遍历每个目标的信息
        for object_score, object_class, object_box in zip(
                per_image_scores, per_image_classes, per_image_boxes):
            object_score = float(object_score)
            class_idx = int(object_class)
            if self.classes_mapping is not None:
                class_idx = self.classes_mapping[class_idx]
            # We recommend rounding coordinates to the nearest tenth of a pixel
            # to reduce resulting JSON file size.
            object_box = [round(b, 2) for b in object_box.tolist()]

            res = {"image_id": img_id,
                   "category_id": class_idx,
                   "bbox": object_box,
                   "score": round(object_score, 3)}
            self.results.append(res)

    def update(self, targets, outputs):
        if self.iou_type == "bbox":
            self.prepare_for_coco_detection(targets, outputs)
        else:
            raise KeyError(f"not support iou_type: {self.iou_type}")

    def evaluate(self):
        # write predict results into json file
        json_str = json.dumps(self.results, indent=4)
        with open(self.results_file_name, 'w') as json_file:
            json_file.write(json_str)

        # accumulate predictions from all images
        coco_true = self.coco
        coco_pre = coco_true.loadRes(self.results_file_name)

        self.coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType=self.iou_type)

        self.coco_evaluator.evaluate()
        self.coco_evaluator.accumulate()
        print(f"IoU metric: {self.iou_type}")
        self.coco_evaluator.summarize()

        coco_info = self.coco_evaluator.stats.tolist()  # numpy to list
        return coco_info

