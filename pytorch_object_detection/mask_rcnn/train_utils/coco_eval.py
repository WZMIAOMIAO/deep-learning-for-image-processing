import json
import copy

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
from .distributed_utils import all_gather, is_main_process


class DetectionMetric:
    def __init__(self, coco: COCO = None, classes_mapping: dict = None):
        self.coco = copy.deepcopy(coco)
        self.results = []
        self.aggregation_results = None
        self.classes_mapping = classes_mapping
        self.coco_evaluator = None
        self.iou_type = "bbox"

    def update(self, targets, outputs):
        # 遍历每张图像的预测结果
        for target, output in zip(targets, outputs):
            if len(output) == 0:
                continue

            img_id = int(target["image_id"])
            per_image_boxes = output["boxes"]
            # 对于coco_eval, 需要的每个box的数据格式为[x_min, y_min, w, h]
            # 而我们预测的box格式是[x_min, y_min, x_max, y_max]，所以需要转下格式
            per_image_boxes[:, 2:] -= per_image_boxes[:, :2]
            per_image_classes = output["labels"].tolist()
            per_image_scores = output["scores"].tolist()

            # 遍历每个目标的信息
            for object_score, object_class, object_box in zip(
                    per_image_scores, per_image_classes, per_image_boxes):
                object_score = float(object_score)
                class_idx = int(object_class)
                if self.classes_mapping is not None:
                    class_idx = int(self.classes_mapping[str(class_idx)])
                # We recommend rounding coordinates to the nearest tenth of a pixel
                # to reduce resulting JSON file size.
                object_box = [round(b, 2) for b in object_box.tolist()]

                res = {"image_id": img_id,
                       "category_id": class_idx,
                       "bbox": object_box,
                       "score": round(object_score, 3)}
                self.results.append(res)

    def synchronize_results(self):
        # 同步所有进程中的数据
        all_results = all_gather(self.results)

        # 将所有进程上的数据合并到一个list当中
        results = []
        for res in all_results:
            results.extend(res)

        self.aggregation_results = results

        # 主进程上保存即可
        if is_main_process():
            # write predict results into json file
            json_str = json.dumps(results, indent=4)
            with open('eval_det_tmp.json', 'w') as json_file:
                json_file.write(json_str)

    def evaluate(self):
        # 只在主进程上评估即可
        if is_main_process():
            # accumulate predictions from all images
            coco_true = self.coco
            coco_pre = coco_true.loadRes('eval_det_tmp.json')

            self.coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType=self.iou_type)
            self.coco_evaluator.evaluate()
            self.coco_evaluator.accumulate()
            print(f"IoU metric: {self.iou_type}")
            self.coco_evaluator.summarize()

            coco_info = self.coco_evaluator.stats.tolist()  # numpy to list
            return coco_info
        else:
            return None


class SegmentationMetric:
    def __init__(self, coco: COCO = None, classes_mapping: dict = None):
        self.coco = copy.deepcopy(coco)
        self.results = []
        self.aggregation_results = None
        self.classes_mapping = classes_mapping
        self.coco_evaluator = None
        self.iou_type = "segm"

    def update(self, targets, outputs):
        # 遍历每张图像的预测结果
        for target, output in zip(targets, outputs):
            if len(output) == 0:
                continue

            img_id = int(target["image_id"])
            per_image_masks = output["masks"]
            per_image_classes = output["labels"].tolist()
            per_image_scores = output["scores"].tolist()

            masks = per_image_masks > 0.5

            # 遍历每个目标的信息
            for mask, label, score in zip(masks, per_image_classes, per_image_scores):
                rle = mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")

                class_idx = int(label)
                if self.classes_mapping is not None:
                    class_idx = int(self.classes_mapping[str(class_idx)])

                res = {"image_id": img_id,
                       "category_id": class_idx,
                       "segmentation": rle,
                       "score": round(score, 3)}
                self.results.append(res)

    def synchronize_results(self):
        # 同步所有进程中的数据
        all_results = all_gather(self.results)

        # 将所有进程上的数据合并到一个list当中
        results = []
        for res in all_results:
            results.extend(res)

        self.aggregation_results = results

        # 只在主进程上保存即可
        if is_main_process():
            # write predict results into json file
            json_str = json.dumps(results, indent=4)
            with open('eval_seg_tmp.json', 'w') as json_file:
                json_file.write(json_str)

    def evaluate(self):
        # 只在主进程上评估即可
        if is_main_process():
            # accumulate predictions from all images
            coco_true = self.coco
            coco_pre = coco_true.loadRes('eval_seg_tmp.json')

            self.coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType=self.iou_type)
            self.coco_evaluator.evaluate()
            self.coco_evaluator.accumulate()
            print(f"IoU metric: {self.iou_type}")
            self.coco_evaluator.summarize()

            coco_info = self.coco_evaluator.stats.tolist()  # numpy to list
            return coco_info
        else:
            return None
