import torch
import numpy as np

from model import shufflenet_v2_x1_0


def main():

    m = shufflenet_v2_x1_0()
    m_info = [(i.name.replace(":0", ""), list(i.shape))
              for i in m.weights]

    weights_path = './shufflenetv2_x1.pth'
    weights_dict = torch.load(weights_path)
    new_weights_dict = dict()
    for key, value in weights_dict.items():
        if "conv1.0.weight" == key:
            value = np.transpose(value.detach().numpy(), (2, 3, 1, 0)).astype(np.float32)
            new_weights_dict["conv1/conv1/kernel"] = value
        elif "conv1.1.weight" == key:
            new_weights_dict["conv1/bn/gamma"] = value
        elif "conv1.1.bias" == key:
            new_weights_dict["conv1/bn/beta"] = value
        elif "conv1.1.running_mean" == key:
            new_weights_dict["conv1/bn/moving_mean"] = value
        elif "conv1.1.running_var" == key:
            new_weights_dict["conv1/bn/moving_variance"] = value
        elif "stage" in key:
            names = key.split(".branch")
            num_stage, num_block = names[0].replace("stage", "").split(".")
            tf_name_prefix = "stage{}_{}/".format(num_stage, num_block)

            torch_name2tf_name = {"1.0.weight": "b1_dw1/dw1/depthwise_kernel",
                                  "1.1.weight": "b1_dw1/bn/gamma",
                                  "1.1.bias": "b1_dw1/bn/beta",
                                  "1.1.running_mean": "b1_dw1/bn/moving_mean",
                                  "1.1.running_var": "b1_dw1/bn/moving_variance",
                                  "1.2.weight": "b1_conv1/conv1/kernel",
                                  "1.3.weight": "b1_conv1/bn/gamma",
                                  "1.3.bias": "b1_conv1/bn/beta",
                                  "1.3.running_mean": "b1_conv1/bn/moving_mean",
                                  "1.3.running_var": "b1_conv1/bn/moving_variance",
                                  "2.0.weight": "b2_conv1/conv1/kernel",
                                  "2.1.weight": "b2_conv1/bn/gamma",
                                  "2.1.bias": "b2_conv1/bn/beta",
                                  "2.1.running_mean": "b2_conv1/bn/moving_mean",
                                  "2.1.running_var": "b2_conv1/bn/moving_variance",
                                  "2.3.weight": "b2_dw1/dw1/depthwise_kernel",
                                  "2.4.weight": "b2_dw1/bn/gamma",
                                  "2.4.bias": "b2_dw1/bn/beta",
                                  "2.4.running_mean": "b2_dw1/bn/moving_mean",
                                  "2.4.running_var": "b2_dw1/bn/moving_variance",
                                  "2.5.weight": "b2_conv2/conv1/kernel",
                                  "2.6.weight": "b2_conv2/bn/gamma",
                                  "2.6.bias": "b2_conv2/bn/beta",
                                  "2.6.running_mean": "b2_conv2/bn/moving_mean",
                                  "2.6.running_var": "b2_conv2/bn/moving_variance"}

            tf_name_postfix = torch_name2tf_name[names[1]]
            tf_name = tf_name_prefix + tf_name_postfix

            if len(value.shape) > 1:  # conv or dwconv
                if "dw" in tf_name:
                    value = np.transpose(value.detach().numpy(), (2, 3, 0, 1)).astype(np.float32)
                else:
                    value = np.transpose(value.detach().numpy(), (2, 3, 1, 0)).astype(np.float32)

            new_weights_dict[tf_name] = value

        elif "conv5.0.weight" == key:
            value = np.transpose(value.detach().numpy(), (2, 3, 1, 0)).astype(np.float32)
            new_weights_dict["conv5/conv1/kernel"] = value
        elif "conv5.1.weight" == key:
            new_weights_dict["conv5/bn/gamma"] = value
        elif "conv5.1.bias" == key:
            new_weights_dict["conv5/bn/beta"] = value
        elif "conv5.1.running_mean" == key:
            new_weights_dict["conv5/bn/moving_mean"] = value
        elif "conv5.1.running_var" == key:
            new_weights_dict["conv5/bn/moving_variance"] = value

        elif "fc.weight" == key:
            value = np.transpose(value.detach().numpy(), (1, 0)).astype(np.float32)
            new_weights_dict["fc/kernel"] = value

        elif "fc.bias" == key:
            new_weights_dict["fc/bias"] = value
        else:
            print(key)

    assert len(m_info) == len(new_weights_dict)

    weights_list = []
    for name, shape in m_info:
        assert name in new_weights_dict, "not found key:'{}'".format(name)
        assert tuple(shape) == new_weights_dict[name].shape, \
            "tf shape:'{}', trans shape:'{}'".format(shape,
                                                     new_weights_dict[name].shape)
        weights_list.append(new_weights_dict[name])

    m.set_weights(weights_list)
    m.save_weights("shufflenetv2_x1_0.h5", save_format="h5")


if __name__ == '__main__':
    main()
