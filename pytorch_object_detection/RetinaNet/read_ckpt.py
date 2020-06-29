import tensorflow as tf
import numpy as np
import torch
from collections import OrderedDict


def rename_var(ckpt_path, new_ckpt_path):
    new_weights_dict = {}
    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as sess:
        var_list = tf.train.list_variables(ckpt_path)
        resnet50 = var_list[:265]
        fpn = var_list[265: 291]
        predictor = var_list[291:]

        for var_name_o, shape in resnet50:
            var_name = var_name_o
            var_name = var_name.replace("FeatureExtractor/resnet_v1_50/", "feature_extractor.body.")
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

            # 将tensorflow中的BN参数转换到pytorch中
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

            # print(var_name)
        # torch.save(new_weights_dict, "./re500.pth")

        for var_name_o, shape in fpn:

            var_name = var_name_o
            var_name = var_name.replace("FeatureExtractor/resnet_v1_50/", "feature_extractor.")
            var_name = var_name.replace("/", ".")

            if "projection" in var_name:
                index = var_name.find("projection")
                num = int(var_name[index + 11]) - 1
                var_name = var_name.replace("projection_" + str(num + 1) + ".biases",
                                            "projection_blocks." + str(num) + ".bias")
                var_name = var_name.replace("projection_" + str(num + 1) + ".weights",
                                            "projection_blocks." + str(num) + ".weight")

                var = tf.train.load_variable(ckpt_path, var_name_o)
                if "weight" in var_name:
                    # 卷积参数weight
                    torch_tensor = torch.tensor(np.transpose(var, (3, 2, 0, 1)).astype(np.float32))
                    new_weights_dict.update({var_name: torch_tensor})
                if "bias" in var_name:
                    # 卷积核参数bias
                    torch_tensor = torch.tensor(var.astype(np.float32))
                    new_weights_dict.update({var_name: torch_tensor})

            if "smoothing" in var_name:
                index = var_name.find("smoothing")
                num = int(var_name[index + 10]) - 1
                var_name = var_name.replace("smoothing_" + str(num + 1) + ".weights",
                                            "smoothing_blocks." + str(num) + "." + "0" + ".weight")
                var_name = var_name.replace("smoothing_" + str(num + 1) + ".BatchNorm.beta",
                                            "smoothing_blocks." + str(num) + "." + "1" + ".bias")
                var_name = var_name.replace("smoothing_" + str(num + 1) + ".BatchNorm.gamma",
                                            "smoothing_blocks." + str(num) + "." + "1" + ".weight")
                var_name = var_name.replace("smoothing_" + str(num + 1) + ".BatchNorm.moving_mean",
                                            "smoothing_blocks." + str(num) + "." + "1" + ".running_mean")
                var_name = var_name.replace("smoothing_" + str(num + 1) + ".BatchNorm.moving_variance",
                                            "smoothing_blocks." + str(num) + "." + "1" + ".running_var")

                var = tf.train.load_variable(ckpt_path, var_name_o)
                if "0.weight" in var_name:
                    # 卷积参数weight
                    torch_tensor = torch.tensor(np.transpose(var, (3, 2, 0, 1)).astype(np.float32))
                    new_weights_dict.update({var_name: torch_tensor})
                else:  # 否则对应bn参数
                    torch_tensor = torch.tensor(var.astype(np.float32))
                    new_weights_dict.update({var_name: torch_tensor})

            if "bottom_up_block" in var_name:
                index = var_name.find("bottom_up_block")
                num = int(var_name[index + 15])
                var_name = var_name.replace("bottom_up_block" + str(num) + ".weights",
                                            "extra_blocks." + "bottom_up_block" + str(num) + ".0" + ".weight")
                var_name = var_name.replace("bottom_up_block" + str(num) + ".BatchNorm.beta",
                                            "extra_blocks." + "bottom_up_block" + str(num) + ".1" + ".bias")
                var_name = var_name.replace("bottom_up_block" + str(num) + ".BatchNorm.gamma",
                                            "extra_blocks." + "bottom_up_block" + str(num) + ".1" + ".weight")
                var_name = var_name.replace("bottom_up_block" + str(num) + ".BatchNorm.moving_mean",
                                            "extra_blocks." + "bottom_up_block" + str(num) + ".1" + ".running_mean")
                var_name = var_name.replace("bottom_up_block" + str(num) + ".BatchNorm.moving_variance",
                                            "extra_blocks." + "bottom_up_block" + str(num) + ".1" + ".running_var")

                var = tf.train.load_variable(ckpt_path, var_name_o)
                if "0.weight" in var_name:
                    # 卷积参数weight
                    torch_tensor = torch.tensor(np.transpose(var, (3, 2, 0, 1)).astype(np.float32))
                    new_weights_dict.update({var_name: torch_tensor})
                else:  # 否则对应bn参数
                    torch_tensor = torch.tensor(var.astype(np.float32))
                    new_weights_dict.update({var_name: torch_tensor})

            # print(var_name)

        for var_name_o, shape in predictor:
            if var_name_o in except_list:
                continue
            var_name = var_name_o
            var_name = var_name.replace("WeightSharedConvolutionalBoxPredictor", "predictor")
            var_name = var_name.replace("/", ".")

            if "conv2d" in var_name and "BatchNorm" not in var_name:
                var_name = var_name.replace("BoxPredictionTower", "shared_box_tower_conv")
                var_name = var_name.replace("ClassPredictionTower", "shared_class_tower_conv")
                index = var_name.find("conv2d")
                num = int(var_name[index + 7])
                var_name = var_name.replace(var_name[index:], str(num) + ".weight")

            if "conv2d" in var_name and "BatchNorm" in var_name:
                var_name = var_name.replace("BoxPredictionTower", "unshared_box_tower_bn")
                var_name = var_name.replace("ClassPredictionTower", "unshared_class_tower_bn")
                index1 = var_name.find("conv2d")   # [0, 1, 2, 3]
                index2 = var_name.find("feature")  # [0, 1, 2, 3, 4]
                conv_num = int(var_name[index1 + 7])
                feature_num = int(var_name[index2 + 8])
                if "beta" in var_name:
                    var_name = var_name.replace(var_name[index1:],
                                                str(feature_num) + "." + str(conv_num) + ".bias")
                if "gamma" in var_name:
                    var_name = var_name.replace(var_name[index1:],
                                                str(feature_num) + "." + str(conv_num) + ".weight")
                if "moving_mean" in var_name:
                    var_name = var_name.replace(var_name[index1:],
                                                str(feature_num) + "." + str(conv_num) + ".running_mean")

                if "moving_variance" in var_name:
                    var_name = var_name.replace(var_name[index1:],
                                                str(feature_num) + "." + str(conv_num) + ".running_var")

            if "BoxPredictor" in var_name:
                var_name = var_name.replace("BoxPredictor", "box_predictor")
                if "biases" in var_name:
                    var_name = var_name.replace("biases", "bias")

                if "weights" in var_name:
                    var_name = var_name.replace("weights", "weight")

            if "ClassPredictor" in var_name:
                var_name = var_name.replace("ClassPredictor", "class_predictor")
                if "biases" in var_name:
                    var_name = var_name.replace("biases", "bias")

                if "weights" in var_name:
                    var_name = var_name.replace("weights", "weight")

            var = tf.train.load_variable(ckpt_path, var_name_o)
            if "conv" in var_name or "predictor.weight" in var_name:
                # 卷积参数weight
                torch_tensor = torch.tensor(np.transpose(var, (3, 2, 0, 1)).astype(np.float32))
                new_weights_dict.update({var_name: torch_tensor})
            else:  # 否则对应bn参数以及卷积bias
                torch_tensor = torch.tensor(var.astype(np.float32))
                new_weights_dict.update({var_name: torch_tensor})

            # print(var_name)
        torch.save(new_weights_dict, "./res50_fpn_ssd640.pth")


# except_list = ['global_step', 'resnet_v1_50/mean_rgb', 'resnet_v1_50/logits/biases', 'resnet_v1_50/logits/weights']
except_list = ['global_step']
ckpt_path = 'src/ssd_resnet50_v1_fpn_shared_box_predictor/model.ckpt'
new_ckpt_path = './pretrain_weights.ckpt'
rename_var(ckpt_path, new_ckpt_path)
