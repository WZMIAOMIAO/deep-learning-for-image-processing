from torch import nn, Tensor
import torch
import torch.nn.functional as F
from torch.jit.annotations import Tuple, List, Dict
from collections import OrderedDict


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu6 = nn.ReLU6(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu6(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu6(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu6(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu6 = nn.ReLU6(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu6(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()

        # 遍历模型子模块按顺序存入有序字典
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            # 如果return_layers为空,则停止遍历
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 依次遍历模型的所有子模块，并进行正向传播，
        # 收集layer2, layer3, layer4的输出
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Arguments:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    """

    def __init__(self, in_channels_list, out_channels=256, extra_blocks=None):
        super(FeaturePyramidNetwork, self).__init__()
        # 用来调整resnet输出特征矩阵(layer2,3,4)的channel（kernel_size=1）
        self.projection_blocks = nn.ModuleList()
        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.smoothing_blocks = nn.ModuleList()
        for in_channels in in_channels_list:  # [512, 1024, 2048]
            projection_block = nn.Conv2d(in_channels, out_channels, 1)
            self.projection_blocks.append(projection_block)

            if in_channels in [512, 1024]:  # 只有layer2和layer3才有smoothing_block
                smoothing_block = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                                nn.BatchNorm2d(out_channels),
                                                nn.ReLU6(inplace=True))
                self.smoothing_blocks.append(smoothing_block)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks

    def get_result_from_projection_blocks(self, x, idx):
        # type: (Tensor, int)
        """
        This is equivalent to self.projection_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = 0
        for m in self.projection_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.projection_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_smoothing_blocks(self, x, idx):
        # type: (Tensor, int)
        """
        This is equivalent to self.smoothing_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = 0
        for m in self.smoothing_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.smoothing_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x):
        # type: (Dict[str, Tensor])
        """
        Computes the FPN for a set of feature maps.
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        # 将resnet layer4的channel调整到指定的out_channels
        last_projection = self.projection_blocks[-1](x[-1])

        # result中保存着每个预测特征层
        results = []
        results.append(last_projection)
        # # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        # results.append(self.smoothing_blocks[-1](last_inner))

        # [1, 0]
        for idx in range(len(x)-2, -1, -1):  # 对layer2和layer3进行projection后的特征矩阵进行upsample和smoothing
            projection_lateral = self.get_result_from_projection_blocks(x[idx], idx)
            feat_shape = projection_lateral.shape[-2:]  # [h, w]
            # 将上一层的特征矩阵上采样到当前层大小
            inner_top_down = F.interpolate(last_projection, size=feat_shape, mode="nearest")
            # add
            last_projection = projection_lateral + inner_top_down
            last_projection_smoothing = self.get_result_from_smoothing_blocks(last_projection, idx)
            results.insert(0, last_projection_smoothing)

        # 在layer4对应的预测特征层基础上生成预测特征矩阵P6和P7
        if self.extra_blocks is not None:
            results = self.extra_blocks(results)

        # make it back an OrderedDict
        # out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return results


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.

    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.

    Arguments:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super(BackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,   # [512, 1024, 2048]
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(in_channels=out_channels),
            )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


class LastLevelMaxPool(nn.Module):
    """
    Applies a max_pool2d on top of the last feature map
    """
    def __init__(self, in_channels=256):
        super(LastLevelMaxPool, self).__init__()
        self.bottom_up_block5 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False),
                                              nn.BatchNorm2d(in_channels),
                                              nn.ReLU6(inplace=True))
        self.bottom_up_block6 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False),
                                              nn.BatchNorm2d(in_channels),
                                              nn.ReLU6(inplace=True))

    def forward(self, x):
        # type: (List[Tensor])
        out_middle = self.bottom_up_block5(x[-1])
        x.append(out_middle)

        out_middle = self.bottom_up_block6(out_middle)
        x.append(out_middle)
        return x


def resnet50_fpn_backbone():
    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3], include_top=False)

    return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}
    in_channels_list = [512, 1024, 2048]
    out_channels = 256
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels)
