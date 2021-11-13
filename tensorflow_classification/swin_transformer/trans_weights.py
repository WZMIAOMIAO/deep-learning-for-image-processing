import torch
from model import *


def main(weights_path: str,
         model_name: str,
         model: tf.keras.Model):
    var_dict = {v.name.split(':')[0]: v for v in model.weights}

    weights_dict = torch.load(weights_path, map_location="cpu")["model"]
    w_dict = {}
    for k, v in weights_dict.items():
        if "patch_embed" in k:
            k = k.replace(".", "/")
            if "proj" in k:
                k = k.replace("proj/weight", "proj/kernel")
                if len(v.shape) > 1:
                    # conv weights
                    v = np.transpose(v.numpy(), (2, 3, 1, 0)).astype(np.float32)
                    w_dict[k] = v
                else:
                    # bias
                    w_dict[k] = v
            elif "norm" in k:
                k = k.replace("weight", "gamma").replace("bias", "beta")
                w_dict[k] = v
        elif "layers" in k:
            k = k.replace("layers", "layer")
            split_k = k.split(".")
            layer_id = split_k[0] + split_k[1]
            if "block" in k:
                split_k[2] = "block"
                black_id = split_k[2] + split_k[3]
                k = "/".join([layer_id, black_id, *split_k[4:]])
                if "attn" in k or "mlp" in k:
                    k = k.replace("weight", "kernel")
                    if "kernel" in k:
                        v = np.transpose(v.numpy(), (1, 0)).astype(np.float32)
                elif "norm" in k:
                    k = k.replace("weight", "gamma").replace("bias", "beta")
                w_dict[k] = v
            elif "downsample" in k:
                k = "/".join([layer_id, *split_k[2:]])
                if "reduction" in k:
                    k = k.replace("weight", "kernel")
                    if "kernel" in k:
                        v = np.transpose(v.numpy(), (1, 0)).astype(np.float32)
                elif "norm" in k:
                    k = k.replace("weight", "gamma").replace("bias", "beta")
                w_dict[k] = v
        elif "norm" in k:
            k = k.replace(".", "/").replace("weight", "gamma").replace("bias", "beta")
            w_dict[k] = v
        elif "head" in k:
            k = k.replace(".", "/")
            k = k.replace("weight", "kernel")
            if "kernel" in k:
                v = np.transpose(v.numpy(), (1, 0)).astype(np.float32)
            w_dict[k] = v

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
    model = swin_tiny_patch4_window7_224()
    model.build((1, 224, 224, 3))
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    main(weights_path="./swin_tiny_patch4_window7_224.pth",
         model_name="swin_tiny_patch4_window7_224",
         model=model)

    # model = swin_small_patch4_window7_224()
    # model.build((1, 224, 224, 3))
    # # trained ImageNet-1K
    # # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    # main(weights_path="./swin_small_patch4_window7_224.pth",
    #      model_name="swin_small_patch4_window7_224",
    #      model=model)

    # model = swin_base_patch4_window7_224()
    # model.build((1, 224, 224, 3))
    # # trained ImageNet-1K
    # # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    # main(weights_path="./swin_base_patch4_window7_224.pth",
    #      model_name="swin_base_patch4_window7_224",
    #      model=model)

    # model = swin_base_patch4_window12_384()
    # model.build((1, 384, 384, 3))
    # # trained ImageNet-1K
    # # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
    # main(weights_path="./swin_base_patch4_window12_384.pth",
    #      model_name="swin_base_patch4_window12_384",
    #      model=model)

    # model = swin_base_patch4_window7_224_in22k()
    # model.build((1, 224, 224, 3))
    # # trained ImageNet-22K
    # # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
    # main(weights_path="./swin_base_patch4_window7_224_22k.pth",
    #      model_name="swin_base_patch4_window7_224_22k",
    #      model=model)

    # model = swin_base_patch4_window12_384_in22k()
    # model.build((1, 384, 384, 3))
    # # trained ImageNet-22K
    # # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    # main(weights_path="./swin_base_patch4_window12_384_22k.pth",
    #      model_name="swin_base_patch4_window12_384_22k",
    #      model=model)

    # model = swin_large_patch4_window7_224_in22k()
    # model.build((1, 224, 224, 3))
    # # trained ImageNet-22K
    # # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
    # main(weights_path="./swin_large_patch4_window7_224_22k.pth",
    #      model_name="swin_large_patch4_window7_224_22k",
    #      model=model)

    # model = swin_large_patch4_window12_384_in22k()
    # model.build((1, 384, 384, 3))
    # # trained ImageNet-22K
    # # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    # main(weights_path="./swin_large_patch4_window12_384_22k.pth",
    #      model_name="swin_large_patch4_window12_384_22k",
    #      model=model)
