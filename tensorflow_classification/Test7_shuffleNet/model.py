import tensorflow as tf
from tensorflow.keras import layers, Model


class ConvBNReLU(layers.Layer):
    def __init__(self,
                 filters: int = 1,
                 kernel_size: int = 1,
                 strides: int = 1,
                 padding: str = 'same',
                 **kwargs):
        super(ConvBNReLU, self).__init__(**kwargs)

        self.conv = layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  use_bias=False,
                                  kernel_regularizer=tf.keras.regularizers.l2(4e-5),
                                  name="conv1")
        self.bn = layers.BatchNormalization(momentum=0.9, name="bn")
        self.relu = layers.ReLU()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x


class DWConvBN(layers.Layer):
    def __init__(self,
                 kernel_size: int = 3,
                 strides: int = 1,
                 padding: str = 'same',
                 **kwargs):
        super(DWConvBN, self).__init__(**kwargs)
        self.dw_conv = layers.DepthwiseConv2D(kernel_size=kernel_size,
                                              strides=strides,
                                              padding=padding,
                                              use_bias=False,
                                              kernel_regularizer=tf.keras.regularizers.l2(4e-5),
                                              name="dw1")
        self.bn = layers.BatchNormalization(momentum=0.9, name="bn")

    def call(self, inputs, training=None, **kwargs):
        x = self.dw_conv(inputs)
        x = self.bn(x, training=training)
        return x


class ChannelShuffle(layers.Layer):
    def __init__(self, shape, groups: int = 2, **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        batch_size, height, width, num_channels = shape
        assert num_channels % 2 == 0
        channel_per_group = num_channels // groups

        # Tuple of integers, does not include the samples dimension (batch size).
        self.reshape1 = layers.Reshape((height, width, groups, channel_per_group))
        self.reshape2 = layers.Reshape((height, width, num_channels))

    def call(self, inputs, **kwargs):
        x = self.reshape1(inputs)
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])
        x = self.reshape2(x)
        return x


class ChannelSplit(layers.Layer):
    def __init__(self, num_splits: int = 2, **kwargs):
        super(ChannelSplit, self).__init__(**kwargs)
        self.num_splits = num_splits

    def call(self, inputs, **kwargs):
        b1, b2 = tf.split(inputs,
                          num_or_size_splits=self.num_splits,
                          axis=-1)
        return b1, b2


def shuffle_block_s1(inputs, output_c: int, stride: int, prefix: str):
    if stride != 1:
        raise ValueError("illegal stride value.")

    assert output_c % 2 == 0
    branch_c = output_c // 2

    x1, x2 = ChannelSplit(name=prefix + "/split")(inputs)

    # main branch
    x2 = ConvBNReLU(filters=branch_c, name=prefix + "/b2_conv1")(x2)
    x2 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b2_dw1")(x2)
    x2 = ConvBNReLU(filters=branch_c, name=prefix + "/b2_conv2")(x2)

    x = layers.Concatenate(name=prefix + "/concat")([x1, x2])
    x = ChannelShuffle(x.shape, name=prefix + "/channelshuffle")(x)

    return x


def shuffle_block_s2(inputs, output_c: int, stride: int, prefix: str):
    if stride != 2:
        raise ValueError("illegal stride value.")

    assert output_c % 2 == 0
    branch_c = output_c // 2

    # shortcut branch
    x1 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b1_dw1")(inputs)
    x1 = ConvBNReLU(filters=branch_c, name=prefix + "/b1_conv1")(x1)

    # main branch
    x2 = ConvBNReLU(filters=branch_c, name=prefix + "/b2_conv1")(inputs)
    x2 = DWConvBN(kernel_size=3, strides=stride, name=prefix + "/b2_dw1")(x2)
    x2 = ConvBNReLU(filters=branch_c, name=prefix + "/b2_conv2")(x2)

    x = layers.Concatenate(name=prefix + "/concat")([x1, x2])
    x = ChannelShuffle(x.shape, name=prefix + "/channelshuffle")(x)

    return x


def shufflenet_v2(num_classes: int,
                  input_shape: tuple,
                  stages_repeats: list,
                  stages_out_channels: list):
    img_input = layers.Input(shape=input_shape)
    if len(stages_repeats) != 3:
        raise ValueError("expected stages_repeats as list of 3 positive ints")
    if len(stages_out_channels) != 5:
        raise ValueError("expected stages_out_channels as list of 5 positive ints")

    x = ConvBNReLU(filters=stages_out_channels[0],
                   kernel_size=3,
                   strides=2,
                   name="conv1")(img_input)

    x = layers.MaxPooling2D(pool_size=(3, 3),
                            strides=2,
                            padding='same',
                            name="maxpool")(x)

    stage_name = ["stage{}".format(i) for i in [2, 3, 4]]
    for name, repeats, output_channels in zip(stage_name,
                                              stages_repeats,
                                              stages_out_channels[1:]):
        for i in range(repeats):
            if i == 0:
                x = shuffle_block_s2(x, output_c=output_channels, stride=2, prefix=name + "_{}".format(i))
            else:
                x = shuffle_block_s1(x, output_c=output_channels, stride=1, prefix=name + "_{}".format(i))

    x = ConvBNReLU(filters=stages_out_channels[-1], name="conv5")(x)

    x = layers.GlobalAveragePooling2D(name="globalpool")(x)

    x = layers.Dense(units=num_classes, name="fc")(x)
    x = layers.Softmax()(x)

    model = Model(img_input, x, name="ShuffleNetV2_1.0")

    return model


def shufflenet_v2_x1_0(num_classes=1000, input_shape=(224, 224, 3)):
    # 权重链接: https://pan.baidu.com/s/1M2mp98Si9eT9qT436DcdOw  密码: mhts
    model = shufflenet_v2(num_classes=num_classes,
                          input_shape=input_shape,
                          stages_repeats=[4, 8, 4],
                          stages_out_channels=[24, 116, 232, 464, 1024])
    return model


def shufflenet_v2_x0_5(num_classes=1000, input_shape=(224, 224, 3)):
    model = shufflenet_v2(num_classes=num_classes,
                          input_shape=input_shape,
                          stages_repeats=[4, 8, 4],
                          stages_out_channels=[24, 48, 96, 192, 1024])
    return model


def shufflenet_v2_x2_0(num_classes=1000, input_shape=(224, 224, 3)):
    model = shufflenet_v2(num_classes=num_classes,
                          input_shape=input_shape,
                          stages_repeats=[4, 8, 4],
                          stages_out_channels=[24, 244, 488, 976, 2048])
    return model
