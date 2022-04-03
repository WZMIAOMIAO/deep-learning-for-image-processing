from collections import OrderedDict
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign

from .faster_rcnn_framework import FasterRCNN


class MaskRCNN(FasterRCNN):
    """
        Implements Mask R-CNN.

        The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
        image, and should be in 0-1 range. Different images can have different sizes.

        The behavior of the model changes depending if it is in training or evaluation mode.

        During training, the model expects both the input tensors, as well as a targets (list of dictionary),
        containing:
            - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
              ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            - labels (Int64Tensor[N]): the class label for each ground-truth box
            - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

        The model returns a Dict[Tensor] during training, containing the classification and regression
        losses for both the RPN and the R-CNN, and the mask loss.

        During inference, the model requires only the input tensors, and returns the post-processed
        predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
        follows:
            - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
              ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
            - labels (Int64Tensor[N]): the predicted labels for each image
            - scores (Tensor[N]): the scores or each prediction
            - masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range. In order to
              obtain the final segmentation masks, the soft masks can be thresholded, generally
              with a value of 0.5 (mask >= 0.5)

        Args:
            backbone (nn.Module): the network used to compute the features for the model.
                It should contain a out_channels attribute, which indicates the number of output
                channels that each feature map has (and it should be the same for all feature maps).
                The backbone should return a single Tensor or and OrderedDict[Tensor].
            num_classes (int): number of output classes of the model (including the background).
                If box_predictor is specified, num_classes should be None.
            min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
            max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
            image_mean (Tuple[float, float, float]): mean values used for input normalization.
                They are generally the mean values of the dataset on which the backbone has been trained
                on
            image_std (Tuple[float, float, float]): std values used for input normalization.
                They are generally the std values of the dataset on which the backbone has been trained on
            rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
                maps.
            rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
            rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
            rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
            rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
            rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
            rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
            rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
                considered as positive during training of the RPN.
            rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
                considered as negative during training of the RPN.
            rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
                for computing the loss
            rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
                of the RPN
            rpn_score_thresh (float): during inference, only return proposals with a classification score
                greater than rpn_score_thresh
            box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
                the locations indicated by the bounding boxes
            box_head (nn.Module): module that takes the cropped feature maps as input
            box_predictor (nn.Module): module that takes the output of box_head and returns the
                classification logits and box regression deltas.
            box_score_thresh (float): during inference, only return proposals with a classification score
                greater than box_score_thresh
            box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
            box_detections_per_img (int): maximum number of detections per image, for all classes.
            box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
                considered as positive during training of the classification head
            box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
                considered as negative during training of the classification head
            box_batch_size_per_image (int): number of proposals that are sampled during training of the
                classification head
            box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
                of the classification head
            bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
                bounding boxes
            mask_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
                 the locations indicated by the bounding boxes, which will be used for the mask head.
            mask_head (nn.Module): module that takes the cropped feature maps as input
            mask_predictor (nn.Module): module that takes the output of the mask_head and returns the
                segmentation mask logits

        """

    def __init__(
            self,
            backbone,
            num_classes=None,
            # transform parameters
            min_size=800,
            max_size=1333,
            image_mean=None,
            image_std=None,
            # RPN parameters
            rpn_anchor_generator=None,
            rpn_head=None,
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=0.7,
            rpn_fg_iou_thresh=0.7,
            rpn_bg_iou_thresh=0.3,
            rpn_batch_size_per_image=256,
            rpn_positive_fraction=0.5,
            rpn_score_thresh=0.0,
            # Box parameters
            box_roi_pool=None,
            box_head=None,
            box_predictor=None,
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            box_detections_per_img=100,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=512,
            box_positive_fraction=0.25,
            bbox_reg_weights=None,
            # Mask parameters
            mask_roi_pool=None,
            mask_head=None,
            mask_predictor=None,
    ):

        if not isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"mask_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(mask_roi_pool)}"
            )

        if num_classes is not None:
            if mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")

        out_channels = backbone.out_channels

        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

        super().__init__(
            backbone,
            num_classes,
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # RPN-specific parameters
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
        )

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor


class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Args:
            in_channels (int): number of input channels
            layers (tuple): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels

        for layer_idx, layers_features in enumerate(layers, 1):
            d[f"mask_fcn{layer_idx}"] = nn.Conv2d(next_feature,
                                                  layers_features,
                                                  kernel_size=3,
                                                  stride=1,
                                                  padding=dilation,
                                                  dilation=dilation)
            d[f"relu{layer_idx}"] = nn.ReLU(inplace=True)
            next_feature = layers_features

        super().__init__(d)
        # initial params
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(OrderedDict([
            ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0))
        ]))
        # initial params
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
