import tensorflow as tf
from tensorflow.keras import Model, layers, initializers
import numpy as np


class PatchEmbed(layers.Layer):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = (patch_size, patch_size)
        self.norm = norm_layer() if norm_layer else layers.Activation('linear')

        self.proj = layers.Conv2D(filters=embed_dim, kernel_size=patch_size,
                                  strides=patch_size, padding='SAME',
                                  kernel_initializer=initializers.LecunNormal(),
                                  bias_initializer=initializers.Zeros())

    def call(self, x, **kwargs):
        _, H, W, _ = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            paddings = tf.constant([[0, 0],
                                    [0, self.patch_size[0] - H % self.patch_size[0]],
                                    [0, self.patch_size[1] - W % self.patch_size[1]]])
            x = tf.pad(x, paddings)

        # 下采样patch_size倍
        x = self.proj(x)
        B, H, W, C = x.shape
        # [B, H, W, C] -> [B, H*W, C]
        x = tf.reshape(x, [B, -1, C])
        x = self.norm(x)
        return x, H, W


def window_partition(x, window_size: int):
    """
        将feature map按照window_size划分成一个个没有重叠的window
        Args:
            x: (B, H, W, C)
            window_size (int): window size(M)

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
    B, H, W, C = x.shape
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    # transpose: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # reshape: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # reshape: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # reshape: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, -1])
    return x


class PatchMerging(layers.Layer):
    def __init__(self, dim: int, norm_layer=layers.LayerNormalization):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = layers.Dense(2*dim, use_bias=False)
        self.norm = norm_layer(epsilon=1e-6)

    def call(self, x, H, W):
        """
        x: [B, H*W, C]
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = tf.reshape(x, [B, H, W, C])
        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 != 0) or (W % 2 != 0)
        if pad_input:
            paddings = tf.constant([[0, 0],
                                    [0, 1],
                                    [0, 1]])
            x = tf.pad(x, paddings)

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = tf.concat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = tf.reshape(x, [B, -1, 4*C])  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class MLP(layers.Layer):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    k_ini = initializers.GlorotUniform()
    b_ini = initializers.RandomNormal(stddev=1e-6)

    def __init__(self, in_features, mlp_ratio=4.0, drop=0., name=None):
        super(MLP, self).__init__(name=name)
        self.fc1 = layers.Dense(int(in_features * mlp_ratio), name="Dense_0",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.act = layers.Activation("gelu")
        self.fc2 = layers.Dense(in_features, name="Dense_1",
                                kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.drop = layers.Dropout(drop)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x

