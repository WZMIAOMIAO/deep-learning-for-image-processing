from torch import nn, Tensor
import torch
from torch.jit.annotations import Optional, List, Dict, Tuple, Module
from src.utils import dboxes640_coco, Encoder, PostProcess
from src.loss import Loss


class RetinaNet640(nn.Module):
    def __init__(self, backbone=None, num_classes=21):
        super(RetinaNet640, self).__init__()
        if backbone is None:
            raise Exception("backbone is None")
        if not hasattr(backbone, "out_channels"):
            raise Exception("the backbone not has attribute: out_channel")
        self.feature_extractor = backbone

        self.num_classes = num_classes
        # out_channels = [1024, 512, 512, 256, 256, 256] for resnet50
        self.predictor = Predictor(num_features=5, in_channels=256, num_layers_before_predictor=4,
                                   num_classes=num_classes, num_boxes=6)

        default_box = dboxes640_coco()
        self.compute_loss = Loss(default_box)
        self.encoder = Encoder(default_box)
        self.postprocess = PostProcess(default_box)

    def forward(self, image, targets=None):
        x = self.feature_extractor(image)

        pre_box, pre_class = self.predictor(x)

        # For RetinaNet, shall return nbatch x 76725 x {nlabels, nlocs} results
        # 80x80x9 + 40x40x9 + 20x20x9 + 10x10x9 + 5x5x9 = 76725

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            # bboxes_out (Tensor 76725 x 4), labels_out (Tensor 76725)
            bboxes_out = targets['boxes']
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            # print(bboxes_out.is_contiguous())
            labels_out = targets['labels']
            # print(labels_out.is_contiguous())

            # ploc, plabel, gloc, glabel
            loss = self.compute_loss(pre_box, pre_class, bboxes_out, labels_out)
            return {"total_losses": loss}

        # 将预测回归参数叠加到default box上得到最终预测box，并执行非极大值抑制虑除重叠框
        # results = self.encoder.decode_batch(locs, confs)
        results = self.postprocess(pre_box, pre_class)
        return results


class Predictor(nn.Module):
    def __init__(self, num_features=5, in_channels=256,
                 num_layers_before_predictor=4, num_classes=21, num_boxes=6):
        super(Predictor, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers_before_predictor = num_layers_before_predictor
        self.shared_box_tower_conv = nn.ModuleList()
        self.shared_class_tower_conv = nn.ModuleList()
        # 构建共享的predictor tower权重
        for i in range(num_layers_before_predictor):  # [0, 1, 2, 3]
            self.shared_box_tower_conv.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False))
            self.shared_class_tower_conv.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False))

        # 构建非共享的bn以及activation
        self.unshared_box_tower_bn = nn.ModuleList()
        self.unshared_class_tower_bn = nn.ModuleList()
        self.unshared_box_tower_relu6 = nn.ModuleList()
        self.unshared_class_tower_relu6 = nn.ModuleList()
        # 每个预测特征层的bn和activation都不共享
        for i in range(num_features):  # [0, 1, 2, 3, 4]
            box_bn_every_layer = nn.ModuleList()
            box_relu6_every_layer = nn.ModuleList()
            class_bn_every_layer = nn.ModuleList()
            class_relu6_every_layer = nn.ModuleList()

            for j in range(num_layers_before_predictor):  # [0, 1, 2, 3]
                box_bn_every_layer.append(nn.BatchNorm2d(in_channels))
                box_relu6_every_layer.append(nn.ReLU6(inplace=True))

                class_bn_every_layer.append(nn.BatchNorm2d(in_channels))
                class_relu6_every_layer.append(nn.ReLU6(inplace=True))

            self.unshared_box_tower_bn.append(box_bn_every_layer)
            self.unshared_box_tower_relu6.append(box_relu6_every_layer)

            self.unshared_class_tower_bn.append(class_bn_every_layer)
            self.unshared_class_tower_relu6.append(class_relu6_every_layer)

        self.box_predictor = nn.Conv2d(in_channels, num_boxes * 4, kernel_size=3, stride=1, padding=1)
        self.class_predictor = nn.Conv2d(in_channels, num_boxes * num_classes, kernel_size=3, stride=1, padding=1)
        self._init_weights()

    def _init_weights(self):
        layers = [*self.shared_box_tower_conv, *self.shared_class_tower_conv,
                  self.box_predictor]
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # 参考tf初始化方法
        nn.init.normal_(self.class_predictor.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.class_predictor.bias, -4.6)

    def forward(self, features):
        class_outputs = torch.jit.annotate(List[Tensor], [])
        box_outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(self.num_features):  # [0, 1, 2, 3, 4]
            feature = features[i]
            box_output = feature
            class_output = feature
            batch = feature.size(0)

            # 通过predictor tower层
            for j in range(self.num_layers_before_predictor):  # [0, 1, 2, 3]
                box_output = self.shared_box_tower_conv[j](box_output)              # conv2d_j: conv2d
                box_output = self.unshared_box_tower_bn[i][j](box_output)           # conv2d_j: bn
                box_output = self.unshared_box_tower_relu6[i][j](box_output)        # conv2d_j: RELU6

                class_output = self.shared_class_tower_conv[j](class_output)        # conv2d_j: conv2d
                class_output = self.unshared_class_tower_bn[i][j](class_output)     # conv2d_j: bn
                class_output = self.unshared_class_tower_relu6[i][j](class_output)  # conv2d_j: RELU6

            # 预测box回归信息
            # # [batch, n*4, feat_size, feat_size] -> [batch, 4, -1]
            box_outputs.append(self.box_predictor(box_output).view(batch, 4, -1))
            # 预测class类别信息
            class_outputs.append(self.class_predictor(class_output).view(batch, self.num_classes, -1))

        return torch.cat(box_outputs, 2).contiguous(), torch.cat(class_outputs, 2).contiguous()
