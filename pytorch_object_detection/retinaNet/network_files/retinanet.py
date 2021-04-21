import math
import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Union

import torch
from torch import nn, Tensor

from . import det_utils
from .anchor_utils import AnchorsGenerator
from . import boxes as box_ops
from .losses import sigmoid_focal_loss
from .transform import GeneralizedRCNNTransform


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01):
        super(RetinaNetClassificationHead, self).__init__()

        # class subnet是由四个3x3的卷积层(激活函数为ReLU) + 一个3x3的卷积层(分类器)
        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)

        # initial weights
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def compute_loss(self,
                     targets: List[Dict[str, Tensor]],
                     head_outputs: Dict[str, Tensor],
                     matched_idxs: List[Tensor]) -> Tensor:
        losses = []
        cls_logits = head_outputs["cls_logits"]
        for targets_per_img, cls_logits_per_img, matched_idxs_per_img in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            # 找出所有前景目标
            foreground_idxs_per_img = torch.ge(matched_idxs_per_img, 0)  # ge: >=
            num_foreground = foreground_idxs_per_img.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_img)
            gt_classes_target[
                foreground_idxs_per_img,
                targets_per_img["labels"][matched_idxs_per_img[foreground_idxs_per_img]]
            ] = 1.0

            # find indices for which anchors should be ignored
            # 忽略iou在[0.4, 0.5)之间的anchors
            valid_idxs_per_img = torch.ne(matched_idxs_per_img, self.BETWEEN_THRESHOLDS)  # ne: !=

            # compute the classification loss
            losses.append(sigmoid_focal_loss(
                cls_logits_per_img[valid_idxs_per_img],
                gt_classes_target[valid_idxs_per_img],
                reduction="sum"
            ) / max(1, num_foreground))  # 注意这里除以的是正样本的个数

        # len(targets): batch_size
        return _sum(losses) / len(targets)

    def forward(self, x: Tensor) -> Tensor:
        all_cls_logits = []

        # 遍历每个预测特征层
        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            # [N, A, K, H, W] -> [N, H, W, A, K]
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            # [N, H, W, A, K] -> [N, HWA, K]
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
    }

    def __init__(self, in_channels, num_anchors):
        super(RetinaNetRegressionHead, self).__init__()

        # box subnet是由四个3x3的卷积层(激活函数为ReLU) + 一个3x3的卷积层(边界框回归器)
        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        # initial weights
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.bbox_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def compute_loss(self,
                     targets: List[Dict[str, Tensor]],
                     head_outputs: Dict[str, Tensor],
                     anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Tensor:
        losses = []

        bbox_regression = head_outputs["bbox_regression"]
        for targets_per_img, bbox_regression_per_img, anchors_per_img, matched_idxs_per_img in \
                zip(targets, bbox_regression, anchors, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_img = torch.where(torch.ge(matched_idxs_per_img, 0))[0]  # ge: >=
            num_foreground = foreground_idxs_per_img.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_img = targets_per_img["boxes"][matched_idxs_per_img[foreground_idxs_per_img]]
            bbox_regression_per_img = bbox_regression_per_img[foreground_idxs_per_img, :]
            anchors_per_img = anchors_per_img[foreground_idxs_per_img, :]

            # compute the regression targets
            targets_regression = self.bbox_coder.encode_single(matched_gt_boxes_per_img, anchors_per_img)

            # compute the box regression loss
            losses.append(torch.nn.functional.l1_loss(
                bbox_regression_per_img,
                targets_regression,
                reduction="sum"
            ) / max(1, num_foreground))

        return _sum(losses) / max(1, len(targets))

    def forward(self, x: List[Tensor]) -> Tensor:
        all_bbox_regression = []

        # 遍历每个预测特征层
        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            # [N, 4 * A, H, W] -> [N, A, 4, H, W]
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            # [N, A, 4, H, W] -> [N, H, W, A, 4]
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            # [N, H, W, A, 4] -> [N, HWA, 4]
            bbox_regression = bbox_regression.reshape(N, -1, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)


class RetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes):
        super(RetinaNetHead, self).__init__()
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)

    def compute_loss(self,
                     targets: List[Dict[str, Tensor]],
                     head_outputs: Dict[str, Tensor],
                     anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        return {
            "classification": self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            "bbox_regression": self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs)
        }

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            "cls_logits": self.classification_head(x),
            "bbox_regression": self.regression_head(x)
        }


class RetinaNet(nn.Module):
    """
    Implements RetinaNet.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (excluding the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        topk_candidates (int): Number of best detections to keep before NMS.
    """

    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
    }

    def __init__(self, backbone, num_classes,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # Anchor parameters
                 anchor_generator=None, head=None,
                 proposal_matcher=None,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=100,
                 fg_iou_thresh=0.5, bg_iou_thresh=0.4,
                 topk_candidates=1000):
        super(RetinaNet, self).__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        self.backbone = backbone

        assert isinstance(anchor_generator, (AnchorsGenerator, type(None)))

        if anchor_generator is None:
            # 原论文中说在每个预测特征层上除了使用给定的尺度x外，还要额外添加x*2^(1/3)和x*2^(2/3)这两个尺度
            # 五个预测特征层采用的原始尺度分别为32， 64， 128， 256， 512
            # 注意尺度和面积的关系，面积=尺度^2
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                                 for x in [32, 64, 128, 256, 512])
            # 对于每个预测特征层上anchors，都会使用三种比例
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorsGenerator(anchor_sizes, aspect_ratios)

        self.anchor_generator = anchor_generator

        if head is None:
            head = RetinaNetHead(backbone.out_channels,   # in_channels
                                 anchor_generator.num_anchors_per_location()[0],  # num_anchors
                                 num_classes)  # num_classes
        self.head = head

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_img, targets_per_img in zip(anchors, targets):
            if targets_per_img["boxes"].numel() == 0:
                matched_idxs.append(torch.full((anchors_per_img.size(0),), -1, dtype=torch.int64))
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_img["boxes"], anchors_per_img)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(self, head_output, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_output["cls_logits"]
        box_regression = head_output["bbox_regression"]

        num_img = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_img):
            box_regression_per_img = [br[index] for br in box_regression]
            logits_per_img = [cl[index] for cl in class_logits]
            anchors_per_img, image_shape = anchors[index], image_shapes[index]

            img_boxes = []
            img_scores = []
            img_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in \
                    zip(box_regression_per_img, logits_per_img, anchors_per_img):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                # 移除低概率的目标
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = torch.gt(scores_per_level, self.score_thresh)  # gt: >
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                # 在每个level上只取前topk个目标
                num_topk = min(self.topk_candidates, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = topk_idxs // num_classes
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(box_regression_per_level[anchor_idxs],
                                                               anchors_per_level[anchor_idxs])
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                img_boxes.append(boxes_per_level)
                img_scores.append(scores_per_level)
                img_labels.append(labels_per_level)

            img_boxes = torch.cat(img_boxes, dim=0)
            img_scores = torch.cat(img_scores, dim=0)
            img_labels = torch.cat(img_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(img_boxes, img_scores, img_labels, self.nms_thresh)
            keep = keep[:self.detections_per_img]

            detections.append({
                "boxes": img_boxes[keep],
                "scores": img_scores[keep],
                "labels": img_labels[keep]
            })

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            # check targets info
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # get the original images sizes
        original_img_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_img_sizes.append((val[0], val[1]))  # h, w

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None
            losses = self.compute_loss(targets, head_outputs, anchors)
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs["cls_logits"].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_img_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)
