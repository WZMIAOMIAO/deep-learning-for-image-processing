import tensorflow as tf
import numpy as np
import torch
from collections import OrderedDict


def rename_var(ckpt_path, new_ckpt_path):
    new_weights_dict = {}
    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as sess:
        var_list = tf.train.list_variables(ckpt_path)
        resnet50 = var_list[:265]

        for var_name_o, shape in resnet50:
            var_name = var_name_o
            var_name = var_name.replace("FeatureExtractor/resnet_v1_50/", "")
            var_name = var_name.replace("/bottleneck_v1", "")
            var_name = var_name.replace("/", ".")
            var_name = var_name.replace("block", "layer")
            var_name = var_name.replace("shortcut", "downsample")

            index = var_name.find("unit_")
            if index != -1:
                num = int(var_name[index+5]) - 1
                var_name = var_name.replace(var_name[index: index+6], str(num))

            if "conv" in var_name and "BatchNorm" in var_name:
                index = var_name.find("BatchNorm")
                num = var_name[index-2]
                var_name = var_name.replace("conv" + num + ".BatchNorm", "bn" + num)
                var_name = var_name.replace("moving_variance", "running_var")
                var_name = var_name.replace("moving_mean", "running_mean")
                var_name = var_name.replace("gamma", "weight")
                var_name = var_name.replace("beta", "bias")

            if "conv" in var_name and "bn" not in var_name:
                var_name = var_name.replace("weights", "weight")

            if "downsample" in var_name:
                var_name = var_name.replace("downsample.weights", "downsample.0.weight")
                var_name = var_name.replace("downsample.BatchNorm.beta", "downsample.1.bias")
                var_name = var_name.replace("downsample.BatchNorm.gamma", "downsample.1.weight")
                var_name = var_name.replace("downsample.BatchNorm.moving_mean", "downsample.1.running_mean")
                var_name = var_name.replace("downsample.BatchNorm.moving_variance", "downsample.1.running_var")

            # 将tensorflow中的卷积核参数转换到pytorch中
            if "conv" in var_name:
                var = tf.train.load_variable(ckpt_path, var_name_o)
                # tensorflow conv2d kernel: [kernel_height, kernel_width, kernel_channel, kernel_number]
                # pytorch conv2d kernel: [kernel_number, kernel_channel, kernel_height, kernel_width]
                torch_tensor = torch.tensor(np.transpose(var, (3, 2, 0, 1)).astype(np.float32))
                new_weights_dict.update({var_name: torch_tensor})

            # # 将tensorflow中的BN参数转换到pytorch中
            if "bn" in var_name:
                var = tf.train.load_variable(ckpt_path, var_name_o)
                torch_tensor = torch.tensor(var.astype(np.float32))
                new_weights_dict.update({var_name: torch_tensor})

            if "downsample" in var_name:
                if "0.weight" in var_name:  # 对应卷积情况
                    var = tf.train.load_variable(ckpt_path, var_name_o)
                    # tensorflow conv2d kernel: [kernel_height, kernel_width, kernel_channel, kernel_number]
                    # pytorch conv2d kernel: [kernel_number, kernel_channel, kernel_height, kernel_width]
                    torch_tensor = torch.tensor(np.transpose(var, (3, 2, 0, 1)).astype(np.float32))
                    new_weights_dict.update({var_name: torch_tensor})
                else:  # 对应BN情况
                    var = tf.train.load_variable(ckpt_path, var_name_o)
                    torch_tensor = torch.tensor(var.astype(np.float32))
                    new_weights_dict.update({var_name: torch_tensor})

            print(var_name)
        # torch.save(new_weights_dict, "./re500.pth")
        fpn = var_list[265:]
        for var_name, shape in fpn:
            print(var_name)
            if var_name in except_list:
                continue
            # var = tf.train.load_variable(ckpt_path, var_name)
            new_var_name = var_name.replace('resnet_v1_50/', "")
            new_var_name = new_var_name.replace("bottleneck_v1/", "")
            new_var_name = new_var_name.replace("shortcut/weights", "shortcut/conv1/kernel")
            new_var_name = new_var_name.replace("weights", "kernel")
            new_var_name = new_var_name.replace("biases", "bias")



# except_list = ['global_step', 'resnet_v1_50/mean_rgb', 'resnet_v1_50/logits/biases', 'resnet_v1_50/logits/weights']
except_list = ['global_step']
ckpt_path = 'src/ssd_resnet50_v1_fpn_shared_box_predictor/model.ckpt'
new_ckpt_path = './pretrain_weights.ckpt'
rename_var(ckpt_path, new_ckpt_path)
