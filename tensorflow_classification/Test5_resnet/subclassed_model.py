from tensorflow.keras import layers, Model, Sequential


class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="SAME", use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False, **kwargs):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x


class Bottleneck(layers.Layer):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="conv1/BatchNorm")
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                   strides=strides, padding="SAME", name="conv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="conv2/BatchNorm")
        # -----------------------------------------
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3")
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="conv3/BatchNorm")
        # -----------------------------------------
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()

    def call(self, inputs, training=False, **kwargs):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([x, identity])
        x = self.relu(x)

        return x


class ResNet(Model):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.include_top = include_top
        self.conv1 = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding="SAME",
                                   use_bias=False, name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="conv1/BatchNorm")
        self.relu1 = layers.ReLU(name="relu1")
        self.maxpool1 = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool1")

        self.block1 = self._make_layer(block, True, 64, blocks_num[0], name="block1")
        self.block2 = self._make_layer(block, False, 128, blocks_num[1], strides=2, name="block2")
        self.block3 = self._make_layer(block, False, 256, blocks_num[2], strides=2, name="block3")
        self.block4 = self._make_layer(block, False, 512, blocks_num[3], strides=2, name="block4")

        if self.include_top:
            self.avgpool = layers.GlobalAvgPool2D(name="avgpool1")
            self.fc = layers.Dense(num_classes, name="logits")
            self.softmax = layers.Softmax()

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)

        if self.include_top:
            x = self.avgpool(x)
            x = self.fc(x)
            x = self.softmax(x)

        return x

    def _make_layer(self, block, first_block, channel, block_num, name=None, strides=1):
        downsample = None
        if strides != 1 or first_block is True:
            downsample = Sequential([
                layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                              use_bias=False, name="conv1"),
                layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
            ], name="shortcut")

        layers_list = []
        layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))

        for index in range(1, block_num):
            layers_list.append(block(channel, name="unit_" + str(index + 1)))

        return Sequential(layers_list, name=name)


def resnet34(num_classes=1000, include_top=True):
    block = BasicBlock
    block_num = [3, 4, 6, 3]
    return ResNet(block, block_num, num_classes, include_top)


def resnet101(num_classes=1000, include_top=True):
    block = Bottleneck
    blocks_num = [3, 4, 23, 3]
    return ResNet(block, blocks_num, num_classes, include_top)



