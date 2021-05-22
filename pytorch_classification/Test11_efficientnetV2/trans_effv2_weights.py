import tensorflow as tf
import torch
import numpy as np


def main(model_name: str = "efficientnetv2-s",
         tf_weights_path: str = "./efficientnetv2-s/model",
         stage0_num: int = 2,
         fused_conv_num: int = 10):

    except_var = ["global_step"]

    new_weights = {}
    var_list = [i for i in tf.train.list_variables(tf_weights_path) if "Exponential" not in i[0]]
    reader = tf.train.load_checkpoint(tf_weights_path)
    for v in var_list:
        if v[0] in except_var:
            continue
        new_name = v[0].replace(model_name + "/", "").replace("/", ".")

        if "stem" in v[0]:
            new_name = new_name.replace("conv2d.kernel",
                                        "conv.weight")

            new_name = new_name.replace("tpu_batch_normalization.beta",
                                        "bn.bias")
            new_name = new_name.replace("tpu_batch_normalization.gamma",
                                        "bn.weight")
            new_name = new_name.replace("tpu_batch_normalization.moving_mean",
                                        "bn.running_mean")
            new_name = new_name.replace("tpu_batch_normalization.moving_variance",
                                        "bn.running_var")
        elif "head" in v[0]:
            new_name = new_name.replace("conv2d.kernel",
                                        "project_conv.conv.weight")
            new_name = new_name.replace("dense.kernel",
                                        "classifier.weight")
            new_name = new_name.replace("dense.bias",
                                        "classifier.bias")

            new_name = new_name.replace("tpu_batch_normalization.beta",
                                        "project_conv.bn.bias")
            new_name = new_name.replace("tpu_batch_normalization.gamma",
                                        "project_conv.bn.weight")
            new_name = new_name.replace("tpu_batch_normalization.moving_mean",
                                        "project_conv.bn.running_mean")
            new_name = new_name.replace("tpu_batch_normalization.moving_variance",
                                        "project_conv.bn.running_var")
        elif "blocks" in v[0]:
            # e.g. blocks_0.conv2d.kernel -> 0
            blocks_id = new_name.split(".", maxsplit=1)[0].replace("blocks_", "")
            new_name = new_name.replace("blocks_{}".format(blocks_id),
                                        "blocks.{}".format(blocks_id))

            if int(blocks_id) <= stage0_num - 1:  # expansion=1 fused_mbconv
                new_name = new_name.replace("conv2d.kernel",
                                            "project_conv.conv.weight")
                new_name = new_name.replace("tpu_batch_normalization.beta",
                                            "project_conv.bn.bias")
                new_name = new_name.replace("tpu_batch_normalization.gamma",
                                            "project_conv.bn.weight")
                new_name = new_name.replace("tpu_batch_normalization.moving_mean",
                                            "project_conv.bn.running_mean")
                new_name = new_name.replace("tpu_batch_normalization.moving_variance",
                                            "project_conv.bn.running_var")
            else:
                new_name = new_name.replace("blocks.{}.conv2d.kernel".format(blocks_id),
                                            "blocks.{}.expand_conv.conv.weight".format(blocks_id))
                new_name = new_name.replace("tpu_batch_normalization.beta",
                                            "expand_conv.bn.bias")
                new_name = new_name.replace("tpu_batch_normalization.gamma",
                                            "expand_conv.bn.weight")
                new_name = new_name.replace("tpu_batch_normalization.moving_mean",
                                            "expand_conv.bn.running_mean")
                new_name = new_name.replace("tpu_batch_normalization.moving_variance",
                                            "expand_conv.bn.running_var")

                if int(blocks_id) <= fused_conv_num - 1:  # fused_mbconv
                    new_name = new_name.replace("blocks.{}.conv2d_1.kernel".format(blocks_id),
                                                "blocks.{}.project_conv.conv.weight".format(blocks_id))
                    new_name = new_name.replace("tpu_batch_normalization_1.beta",
                                                "project_conv.bn.bias")
                    new_name = new_name.replace("tpu_batch_normalization_1.gamma",
                                                "project_conv.bn.weight")
                    new_name = new_name.replace("tpu_batch_normalization_1.moving_mean",
                                                "project_conv.bn.running_mean")
                    new_name = new_name.replace("tpu_batch_normalization_1.moving_variance",
                                                "project_conv.bn.running_var")
                else:  # mbconv
                    new_name = new_name.replace("blocks.{}.conv2d_1.kernel".format(blocks_id),
                                                "blocks.{}.project_conv.conv.weight".format(blocks_id))

                    new_name = new_name.replace("depthwise_conv2d.depthwise_kernel",
                                                "dwconv.conv.weight")

                    new_name = new_name.replace("tpu_batch_normalization_1.beta",
                                                "dwconv.bn.bias")
                    new_name = new_name.replace("tpu_batch_normalization_1.gamma",
                                                "dwconv.bn.weight")
                    new_name = new_name.replace("tpu_batch_normalization_1.moving_mean",
                                                "dwconv.bn.running_mean")
                    new_name = new_name.replace("tpu_batch_normalization_1.moving_variance",
                                                "dwconv.bn.running_var")

                    new_name = new_name.replace("tpu_batch_normalization_2.beta",
                                                "project_conv.bn.bias")
                    new_name = new_name.replace("tpu_batch_normalization_2.gamma",
                                                "project_conv.bn.weight")
                    new_name = new_name.replace("tpu_batch_normalization_2.moving_mean",
                                                "project_conv.bn.running_mean")
                    new_name = new_name.replace("tpu_batch_normalization_2.moving_variance",
                                                "project_conv.bn.running_var")

                    new_name = new_name.replace("se.conv2d.bias",
                                                "se.conv_reduce.bias")
                    new_name = new_name.replace("se.conv2d.kernel",
                                                "se.conv_reduce.weight")
                    new_name = new_name.replace("se.conv2d_1.bias",
                                                "se.conv_expand.bias")
                    new_name = new_name.replace("se.conv2d_1.kernel",
                                                "se.conv_expand.weight")
        else:
            print("not recognized name: " + v[0])

        var = reader.get_tensor(v[0])
        new_var = var
        if "conv" in new_name and "weight" in new_name and "bn" not in new_name and "dw" not in new_name:
            assert len(var.shape) == 4
            # conv kernel [h, w, c, n] -> [n, c, h, w]
            new_var = np.transpose(var, (3, 2, 0, 1))
        elif "bn" in new_name:
            pass
        elif "dwconv" in new_name and "weight" in new_name:
            # dw_kernel [h, w, n, c] -> [n, c, h, w]
            assert len(var.shape) == 4
            new_var = np.transpose(var, (2, 3, 0, 1))
        elif "classifier" in new_name and "weight" in new_name:
            assert len(var.shape) == 2
            new_var = np.transpose(var, (1, 0))

        new_weights[new_name] = torch.as_tensor(new_var)

    torch.save(new_weights, "pre_" + model_name + ".pth")


if __name__ == '__main__':
    main(model_name="efficientnetv2-s",
         tf_weights_path="./efficientnetv2-s/model",
         stage0_num=2,
         fused_conv_num=10)

    # main(model_name="efficientnetv2-m",
    #      tf_weights_path="./efficientnetv2-m/model",
    #      stage0_num=3,
    #      fused_conv_num=13)

    # main(model_name="efficientnetv2-l",
    #      tf_weights_path="./efficientnetv2-l/model",
    #      stage0_num=4,
    #      fused_conv_num=18)
