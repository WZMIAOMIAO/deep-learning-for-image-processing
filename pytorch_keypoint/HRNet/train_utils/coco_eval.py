import json
import copy

from PIL import Image, ImageDraw
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .distributed_utils import all_gather, is_main_process
from transforms import affine_points


def merge(img_ids, eval_results):
    """将多个进程之间的数据汇总在一起"""
    all_img_ids = all_gather(img_ids)
    all_eval_results = all_gather(eval_results)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_results = []
    for p in all_eval_results:
        merged_eval_results.extend(p)

    merged_img_ids = np.array(merged_img_ids)

    # keep only unique (and in sorted order) images
    # 去除重复的图片索引，多GPU训练时为了保证每个进程的训练图片数量相同，可能将一张图片分配给多个进程
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_results = [merged_eval_results[i] for i in idx]

    return list(merged_img_ids), merged_eval_results


class EvalCOCOMetric:
    def __init__(self,
                 coco: COCO = None,
                 iou_type: str = "keypoints",
                 results_file_name: str = "predict_results.json",
                 classes_mapping: dict = None,
                 threshold: float = 0.2):
        self.coco = copy.deepcopy(coco)
        self.obj_ids = []  # 记录每个进程处理目标(person)的ids
        self.results = []
        self.aggregation_results = None
        self.classes_mapping = classes_mapping
        self.coco_evaluator = None
        assert iou_type in ["keypoints"]
        self.iou_type = iou_type
        self.results_file_name = results_file_name
        self.threshold = threshold

    def plot_img(self, img_path, keypoints, r=3):
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        for i, point in enumerate(keypoints):
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                         fill=(255, 0, 0))
        img.show()

    def prepare_for_coco_keypoints(self, targets, outputs):
        # 遍历每个person的预测结果(注意这里不是每张，一张图片里可能有多个person)
        for target, keypoints, scores in zip(targets, outputs[0], outputs[1]):
            if len(keypoints) == 0:
                continue

            obj_idx = int(target["obj_index"])
            if obj_idx in self.obj_ids:
                # 防止出现重复的数据
                continue

            self.obj_ids.append(obj_idx)
            # self.plot_img(target["image_path"], keypoints)

            mask = np.greater(scores, 0.2)
            if mask.sum() == 0:
                k_score = 0
            else:
                k_score = np.mean(scores[mask])

            keypoints = np.concatenate([keypoints, scores], axis=1)
            keypoints = np.reshape(keypoints, -1)

            # We recommend rounding coordinates to the nearest tenth of a pixel
            # to reduce resulting JSON file size.
            keypoints = [round(k, 2) for k in keypoints.tolist()]

            res = {"image_id": target["image_id"],
                   "category_id": 1,  # person
                   "keypoints": keypoints,
                   "score": target["score"] * k_score}

            self.results.append(res)

    def update(self, targets, outputs):
        if self.iou_type == "keypoints":
            self.prepare_for_coco_keypoints(targets, outputs)
        else:
            raise KeyError(f"not support iou_type: {self.iou_type}")

    def synchronize_results(self):
        # 同步所有进程中的数据
        eval_ids, eval_results = merge(self.obj_ids, self.results)
        self.aggregation_results = {"obj_ids": eval_ids, "results": eval_results}

        # 主进程上保存即可
        if is_main_process():
            # results = []
            # [results.extend(i) for i in eval_results]
            # write predict results into json file
            json_str = json.dumps(eval_results, indent=4)
            with open(self.results_file_name, 'w') as json_file:
                json_file.write(json_str)

    def evaluate(self):
        # 只在主进程上评估即可
        if is_main_process():
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
        else:
            return None
