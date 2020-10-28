from tqdm import tqdm

import torch
import torchvision
import torch.utils.data
from pycocotools.coco import COCO


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    # 遍历dataset中的每张图像
    for img_idx in tqdm(range(len(ds)), desc="loading eval info for coco tools."):
        # find better way to get target
        targets, shapes = ds.coco_index(img_idx)
        # targets: [num_obj, 6] , that number 6 means -> (img_index, obj_index, x, y, w, h)
        img_dict = {}
        img_dict['id'] = img_idx
        img_dict['height'] = shapes[0]
        img_dict['width'] = shapes[1]
        dataset['images'].append(img_dict)

        for obj in targets:
            ann = {}
            ann["image_id"] = img_idx
            # 将相对坐标转为绝对坐标
            # box (x, y, w, h)
            boxes = obj[1:]
            # (x, y, w, h) to (xmin, ymin, w, h)
            boxes[:2] -= 0.5*boxes[2:]
            boxes[[0, 2]] *= img_dict["width"]
            boxes[[1, 3]] *= img_dict["height"]
            boxes = boxes.tolist()

            ann["bbox"] = boxes
            ann["category_id"] = int(obj[0])
            categories.add(int(obj[0]))
            ann["area"] = boxes[2] * boxes[3]
            ann["iscrowd"] = 0
            ann["id"] = ann_id
            dataset["annotations"].append(ann)
            ann_id += 1

    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)
