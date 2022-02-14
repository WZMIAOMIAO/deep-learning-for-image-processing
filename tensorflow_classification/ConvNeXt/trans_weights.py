import torch
from model import *


def transpose_weights(m_type, w_dict, k, v):
    if m_type == "conv":
        if len(v.shape) > 1:
            # conv weights
            v = np.transpose(v.numpy(), (2, 3, 1, 0)).astype(np.float32)
        w_dict[k] = v
    elif m_type == "dwconv":
        if len(v.shape) > 1:
            # dwconv weights
            v = np.transpose(v.numpy(), (2, 3, 0, 1)).astype(np.float32)
        w_dict[k] = v
    elif m_type == "linear":
        if len(v.shape) > 1:
            v = np.transpose(v.numpy(), (1, 0)).astype(np.float32)
        w_dict[k] = v
    elif m_type == "norm":
        w_dict[k] = v
    else:
        ValueError(f"not support type:{m_type}")


def main(weights_path: str,
         model_name: str,
         model: tf.keras.Model):
    var_dict = {v.name.split(':')[0]: v for v in model.weights}

    weights_dict = torch.load(weights_path, map_location="cpu")["model"]
    w_dict = {}
    for k, v in weights_dict.items():
        if "downsample_layers" in k:
            split_k = k.split(".")
            if split_k[1] == "0":
                if split_k[2] == "0":
                    k = "stem/conv2d/" + split_k[-1]
                    k = k.replace("weight", "kernel")
                    transpose_weights("conv", w_dict, k, v)
                else:
                    k = "stem/norm/" + split_k[-1]
                    k = k.replace("weight", "gamma")
                    k = k.replace("bias", "beta")
                    transpose_weights("norm", w_dict, k, v)
            else:
                stage = int(split_k[1]) + 1
                if split_k[2] == "1":
                    k = f"downsample{stage}/conv2d/" + split_k[-1]
                    k = k.replace("weight", "kernel")
                    transpose_weights("conv", w_dict, k, v)
                else:
                    k = f"downsample{stage}/norm/" + split_k[-1]
                    k = k.replace("weight", "gamma")
                    k = k.replace("bias", "beta")
                    transpose_weights("norm", w_dict, k, v)
        elif "stages" in k:
            split_k = k.split(".")
            stage = int(split_k[1]) + 1
            block = int(split_k[2])
            if "dwconv" in k:
                k = f"stage{stage}_block{block}/{split_k[-2]}/{split_k[-1]}"
                k = k.replace("weight", "depthwise_kernel")
                transpose_weights("dwconv", w_dict, k, v)
            elif "pwconv" in k:
                k = f"stage{stage}_block{block}/{split_k[-2]}/{split_k[-1]}"
                k = k.replace("weight", "kernel")
                transpose_weights("linear", w_dict, k, v)
            elif "norm" in k:
                k = f"stage{stage}_block{block}/{split_k[-2]}/{split_k[-1]}"
                k = k.replace("weight", "gamma")
                k = k.replace("bias", "beta")
                transpose_weights("norm", w_dict, k, v)
            elif "gamma" in k:
                k = f"stage{stage}_block{block}/{split_k[-1]}"
                transpose_weights("norm", w_dict, k, v)
            else:
                ValueError(f"unrecognized {k}")
        elif "norm" in k:
            split_k = k.split(".")
            k = f"norm/{split_k[-1]}"
            k = k.replace("weight", "gamma")
            k = k.replace("bias", "beta")
            transpose_weights("norm", w_dict, k, v)
        elif "head" in k:
            split_k = k.split(".")
            k = f"head/{split_k[-1]}"
            k = k.replace("weight", "kernel")
            transpose_weights("linear", w_dict, k, v)
        else:
            ValueError(f"unrecognized {k}")

    for key, var in var_dict.items():
        if key in w_dict:
            if w_dict[key].shape != var.shape:
                msg = "shape mismatch: {}".format(key)
                print(msg)
            else:
                var.assign(w_dict[key], read_value=False)
        else:
            msg = "Not found {} in {}".format(key, weights_path)
            print(msg)

    model.save_weights("./{}.h5".format(model_name))


if __name__ == '__main__':
    model = convnext_tiny(num_classes=1000)
    model.build((1, 224, 224, 3))
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    main(weights_path="./convnext_tiny_1k_224_ema.pth",
         model_name="convnext_tiny_1k_224",
         model=model)

    # model = convnext_small(num_classes=1000)
    # model.build((1, 224, 224, 3))
    # # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    # main(weights_path="./convnext_small_1k_224_ema.pth",
    #      model_name="convnext_small_1k_224",
    #      model=model)

    # model = convnext_base(num_classes=1000)
    # model.build((1, 224, 224, 3))
    # # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # main(weights_path="./convnext_base_1k_224_ema.pth",
    #      model_name="convnext_base_1k_224",
    #      model=model)

    # model = convnext_base(num_classes=21841)
    # model.build((1, 224, 224, 3))
    # # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    # main(weights_path="./convnext_base_22k_224.pth",
    #      model_name="convnext_base_22k_224",
    #      model=model)

    # model = convnext_large(num_classes=1000)
    # model.build((1, 224, 224, 3))
    # # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
    # main(weights_path="./convnext_large_1k_224_ema.pth",
    #      model_name="convnext_large_1k_224",
    #      model=model)

    # model = convnext_large(num_classes=21841)
    # model.build((1, 224, 224, 3))
    # # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
    # main(weights_path="./convnext_large_22k_224.pth",
    #      model_name="convnext_large_22k_224",
    #      model=model)

