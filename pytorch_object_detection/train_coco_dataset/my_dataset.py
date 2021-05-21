import os
import json

import torch
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO


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


class CocoDetection(data.Dataset):
    """`MS Coco Detection <https://cocodataset.org/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, dataset="train", transforms=None):
        super(CocoDetection, self).__init__()
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'
        anno_file = "instances_{}2017.json".format(dataset)
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        self.img_root = os.path.join(root, "{}2017".format(dataset))
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.mode = dataset
        self.transforms = transforms
        self.coco = COCO(self.anno_path)

        if dataset == "train":
            # 获取coco数据索引与类别名称的关系
            # 注意在object80中的索引并不是连续的，虽然只有80个类别，但索引还是按照stuff91来排序的
            coco_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])

            # 将stuff91的类别索引重新编排，从1到80
            coco91to80 = dict([(str(k), idx+1) for idx, (k, _) in enumerate(coco_classes.items())])
            json_str = json.dumps(coco91to80, indent=4)
            with open('coco91_to_80.json', 'w') as json_file:
                json_file.write(json_str)

            # 记录重新编排后的索引以及类别名称关系
            coco80_info = dict([(str(idx+1), v) for idx, (_, v) in enumerate(coco_classes.items())])
            json_str = json.dumps(coco80_info, indent=4)
            with open('coco80_indices.json', 'w') as json_file:
                json_file.write(json_str)
        else:
            # 如果是验证集就直接读取生成好的数据
            coco91to80_path = 'coco91_to_80.json'
            assert os.path.exists(coco91to80_path), "file '{}' does not exist.".format(coco91to80_path)

            coco91to80 = json.load(open(coco91to80_path, "r"))

        self.coco91to80 = coco91to80

        ids = list(sorted(self.coco.imgs.keys()))
        if dataset == "train":
            # 移除没有目标，或者目标面积非常小的数据
            valid_ids = _coco_remove_images_without_annotations(self.coco, ids)
            self.ids = valid_ids
        else:
            self.ids = ids

    def parse_targets(self,
                      img_id: int,
                      coco_targets: list,
                      w: int = None,
                      h: int = None):
        # 只筛选出单个对象的情况
        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
        boxes = []
        for obj in anno:
            if obj["bbox"][2] > 0 and obj["bbox"][3] > 0:
                boxes.append(obj["bbox"])

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        if (w is not None) and (h is not None):
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [self.coco91to80[str(obj["category_id"])] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = torch.tensor([img_id])

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')

        w, h = img.size
        target = self.parse_targets(img_id, coco_target, w, h)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# train = CocoDetection("/data/coco_data/", dataset="train")
# print(len(train))
# t = train[0]
# print(t)