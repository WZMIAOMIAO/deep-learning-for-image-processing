import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers, Model

KERNEL_INITIALIZER = {
    "class_name": "TruncatedNormal",
    "config": {
        "stddev": 0.2
    }
}

BIAS_INITIALIZER = "Zeros"


class Block(layers.Layer):
    """
    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6, name: str = None):
        super().__init__(name=name)
        self.layer_scale_init_value = layer_scale_init_value
        self.dwconv = layers.DepthwiseConv2D(7,
                                             padding="same",
                                             depthwise_initializer=KERNEL_INITIALIZER,
                                             bias_initializer=BIAS_INITIALIZER,
                                             name="dwconv")
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.pwconv1 = layers.Dense(4 * dim,
                                    kernel_initializer=KERNEL_INITIALIZER,
                                    bias_initializer=BIAS_INITIALIZER,
                                    name="pwconv1")
        self.act = layers.Activation("gelu")
        self.pwconv2 = layers.Dense(dim,
                                    kernel_initializer=KERNEL_INITIALIZER,
                                    bias_initializer=BIAS_INITIALIZER,
                                    name="pwconv2")
        self.drop_path = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1)) if drop_rate > 0 else None

    def build(self, input_shape):
        if self.layer_scale_init_value > 0:
            self.gamma = self.add_weight(shape=[input_shape[-1]],
                                         initializer=initializers.Constant(self.layer_scale_init_value),
                                         trainable=True,
                                         dtype=tf.float32,
                                         name="gamma")
        else:
            self.gamma = None

    def call(self, x, training=False):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x, training=training)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        if self.drop_path is not None:
            x = self.drop_path(x, training=training)

        return shortcut + x


class Stem(layers.Layer):
    def __init__(self, dim, name: str = None):
        super().__init__(name=name)
        self.conv = layers.Conv2D(dim,
                                  kernel_size=4,
                                  strides=4,
                                  padding="same",
                                  kernel_initializer=KERNEL_INITIALIZER,
                                  bias_initializer=BIAS_INITIALIZER,
                                  name="conv2d")
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.norm(x, training=training)
        return x


class DownSample(layers.Layer):
    def __init__(self, dim, name: str = None):
        super().__init__(name=name)
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.conv = layers.Conv2D(dim,
                                  kernel_size=2,
                                  strides=2,
                                  padding="same",
                                  kernel_initializer=KERNEL_INITIALIZER,
                                  bias_initializer=BIAS_INITIALIZER,
                                  name="conv2d")

    def call(self, x, training=False):
        x = self.norm(x, training=training)
        x = self.conv(x)
        return x


class ConvNeXt(Model):
    r""" ConvNeXt
        A Tensorflow impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, num_classes: int, depths: list, dims: list, drop_path_rate: float = 0.,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.stem = Stem(dims[0], name="stem")

        cur = 0
        dp_rates = np.linspace(start=0, stop=drop_path_rate, num=sum(depths))
        self.stage1 = [Block(dim=dims[0],
                             drop_rate=dp_rates[cur + i],
                             layer_scale_init_value=layer_scale_init_value,
                             name=f"stage1_block{i}")
                       for i in range(depths[0])]
        cur += depths[0]

        self.downsample2 = DownSample(dims[1], name="downsample2")
        self.stage2 = [Block(dim=dims[1],
                             drop_rate=dp_rates[cur + i],
                             layer_scale_init_value=layer_scale_init_value,
                             name=f"stage2_block{i}")
                       for i in range(depths[1])]
        cur += depths[1]

        self.downsample3 = DownSample(dims[2], name="downsample3")
        self.stage3 = [Block(dim=dims[2],
                             drop_rate=dp_rates[cur + i],
                             layer_scale_init_value=layer_scale_init_value,
                             name=f"stage3_block{i}")
                       for i in range(depths[2])]
        cur += depths[2]

        self.downsample4 = DownSample(dims[3], name="downsample4")
        self.stage4 = [Block(dim=dims[3],
                             drop_rate=dp_rates[cur + i],
                             layer_scale_init_value=layer_scale_init_value,
                             name=f"stage4_block{i}")
                       for i in range(depths[3])]

        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.head = layers.Dense(units=num_classes,
                                 kernel_initializer=KERNEL_INITIALIZER,
                                 bias_initializer=BIAS_INITIALIZER,
                                 name="head")

    def call(self, x, training=False):
        x = self.stem(x, training=training)
        for block in self.stage1:
            x = block(x, training=training)

        x = self.downsample2(x, training=training)
        for block in self.stage2:
            x = block(x, training=training)

        x = self.downsample3(x, training=training)
        for block in self.stage3:
            x = block(x, training=training)

        x = self.downsample4(x, training=training)
        for block in self.stage4:
            x = block(x, training=training)

        x = tf.reduce_mean(x, axis=[1, 2])
        x = self.norm(x, training=training)
        x = self.head(x)
        return x


def convnext_tiny(num_classes: int):
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def convnext_small(num_classes: int):
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def convnext_base(num_classes: int):
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes)
    return model


def convnext_large(num_classes: int):
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes)
    return model


def convnext_xlarge(num_classes: int):
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes)
    return model
