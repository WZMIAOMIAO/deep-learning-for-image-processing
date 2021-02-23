import numpy as np
import torch
import tensorflow as tf

assert tf.version.VERSION >= "2.4.0", "version of tf must greater/equal than 2.4.0"


def main():
    # save pytorch weights path
    save_path = "./efficientnetb0.pth"

    # create keras model and download weights
    # EfficientNetB0, EfficientNetB1, EfficientNetB2, ...
    m = tf.keras.applications.EfficientNetB0()

    weights_dict = dict()
    weights = m.weights[3:]  # delete norm weights
    for weight in weights:
        name = weight.name
        data = weight.numpy()

        if "stem_conv/kernel:0" == name:
            torch_name = "features.stem_conv.0.weight"
            weights_dict[torch_name] = np.transpose(data, (3, 2, 0, 1)).astype(np.float32)
        elif "stem_bn/gamma:0" == name:
            torch_name = "features.stem_conv.1.weight"
            weights_dict[torch_name] = data
        elif "stem_bn/beta:0" == name:
            torch_name = "features.stem_conv.1.bias"
            weights_dict[torch_name] = data
        elif "stem_bn/moving_mean:0" == name:
            torch_name = "features.stem_conv.1.running_mean"
            weights_dict[torch_name] = data
        elif "stem_bn/moving_variance:0" == name:
            torch_name = "features.stem_conv.1.running_var"
            weights_dict[torch_name] = data
        elif "block" in name:
            name = name[5:]  # delete "block" word
            block_index = name[:2]  # 1a, 2a, ...
            name = name[3:]  # delete block_index and "_"
            torch_prefix = "features.{}.block.".format(block_index)

            trans_dict = {"expand_conv/kernel:0": "expand_conv.0.weight",
                          "expand_bn/gamma:0": "expand_conv.1.weight",
                          "expand_bn/beta:0": "expand_conv.1.bias",
                          "expand_bn/moving_mean:0": "expand_conv.1.running_mean",
                          "expand_bn/moving_variance:0": "expand_conv.1.running_var",
                          "dwconv/depthwise_kernel:0": "dwconv.0.weight",
                          "bn/gamma:0": "dwconv.1.weight",
                          "bn/beta:0": "dwconv.1.bias",
                          "bn/moving_mean:0": "dwconv.1.running_mean",
                          "bn/moving_variance:0": "dwconv.1.running_var",
                          "se_reduce/kernel:0": "se.fc1.weight",
                          "se_reduce/bias:0": "se.fc1.bias",
                          "se_expand/kernel:0": "se.fc2.weight",
                          "se_expand/bias:0": "se.fc2.bias",
                          "project_conv/kernel:0": "project_conv.0.weight",
                          "project_bn/gamma:0": "project_conv.1.weight",
                          "project_bn/beta:0": "project_conv.1.bias",
                          "project_bn/moving_mean:0": "project_conv.1.running_mean",
                          "project_bn/moving_variance:0": "project_conv.1.running_var"}

            assert name in trans_dict, "key '{}' not in trans_dict".format(name)
            torch_postfix = trans_dict[name]
            torch_name = torch_prefix + torch_postfix
            if torch_postfix in ["expand_conv.0.weight", "se.fc1.weight", "se.fc2.weight", "project_conv.0.weight"]:
                data = np.transpose(data, (3, 2, 0, 1)).astype(np.float32)
            elif torch_postfix == "dwconv.0.weight":
                data = np.transpose(data, (2, 3, 0, 1)).astype(np.float32)
            weights_dict[torch_name] = data
        elif "top_conv/kernel:0" == name:
            torch_name = "features.top.0.weight"
            weights_dict[torch_name] = np.transpose(data, (3, 2, 0, 1)).astype(np.float32)
        elif "top_bn/gamma:0" == name:
            torch_name = "features.top.1.weight"
            weights_dict[torch_name] = data
        elif "top_bn/beta:0" == name:
            torch_name = "features.top.1.bias"
            weights_dict[torch_name] = data
        elif "top_bn/moving_mean:0" == name:
            torch_name = "features.top.1.running_mean"
            weights_dict[torch_name] = data
        elif "top_bn/moving_variance:0" == name:
            torch_name = "features.top.1.running_var"
            weights_dict[torch_name] = data
        elif "predictions/kernel:0" == name:
            torch_name = "classifier.1.weight"
            weights_dict[torch_name] = np.transpose(data, (1, 0)).astype(np.float32)
        elif "predictions/bias:0" == name:
            torch_name = "classifier.1.bias"
            weights_dict[torch_name] = data
        else:
            raise KeyError("no match key '{}'".format(name))

    for k, v in weights_dict.items():
        weights_dict[k] = torch.as_tensor(v)

    torch.save(weights_dict, save_path)
    print("Conversion complete.")


if __name__ == '__main__':
    main()
