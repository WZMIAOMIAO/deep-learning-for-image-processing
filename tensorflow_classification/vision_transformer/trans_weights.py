from vit_model import *


def main(weights_path: str,
         model_name: str,
         model: tf.keras.Model):
    var_dict = {v.name.split(':')[0]: v for v in model.weights}

    ckpt_dict = np.load(weights_path, allow_pickle=False)
    # keys, values = zip(*list(ckpt_dict.items()))
    w_dict = {}
    for k, v in ckpt_dict.items():
        key_ = k.replace("Transformer/", "").\
            replace("MultiHeadDotProductAttention_1", "MultiHeadAttention").\
            replace("MlpBlock_3", "MlpBlock").\
            replace("posembed_input/pos_embedding", "pos_embed").\
            replace("encoder_norm/bias", "encoder_norm/beta").\
            replace("encoder_norm/scale", "encoder_norm/gamma").\
            replace("LayerNorm_0/bias", "LayerNorm_0/beta").\
            replace("LayerNorm_0/scale", "LayerNorm_0/gamma"). \
            replace("LayerNorm_2/bias", "LayerNorm_1/beta"). \
            replace("LayerNorm_2/scale", "LayerNorm_1/gamma").\
            replace("embedding", "patch_embed/conv2d")
        w_dict[key_] = v

    for i in range(model.depth):
        q_kernel = w_dict.pop("encoderblock_{}/MultiHeadAttention/query/kernel".format(i))
        k_kernel = w_dict.pop("encoderblock_{}/MultiHeadAttention/key/kernel".format(i))
        v_kernel = w_dict.pop("encoderblock_{}/MultiHeadAttention/value/kernel".format(i))
        q_kernel = np.reshape(q_kernel, [q_kernel.shape[0], -1])
        k_kernel = np.reshape(k_kernel, [k_kernel.shape[0], -1])
        v_kernel = np.reshape(v_kernel, [v_kernel.shape[0], -1])
        qkv_kernel = np.concatenate([q_kernel, k_kernel, v_kernel], axis=1)
        w_dict["encoderblock_{}/MultiHeadAttention/qkv/kernel".format(i)] = qkv_kernel

        if model.qkv_bias:
            q_bias = w_dict.pop("encoderblock_{}/MultiHeadAttention/query/bias".format(i))
            k_bias = w_dict.pop("encoderblock_{}/MultiHeadAttention/key/bias".format(i))
            v_bias = w_dict.pop("encoderblock_{}/MultiHeadAttention/value/bias".format(i))
            q_bias = np.reshape(q_bias, [-1])
            k_bias = np.reshape(k_bias, [-1])
            v_bias = np.reshape(v_bias, [-1])
            qkv_bias = np.concatenate([q_bias, k_bias, v_bias], axis=0)
            w_dict["encoderblock_{}/MultiHeadAttention/qkv/bias".format(i)] = qkv_bias

        out_kernel = w_dict["encoderblock_{}/MultiHeadAttention/out/kernel".format(i)]
        out_kernel = np.reshape(out_kernel, [-1, out_kernel.shape[-1]])
        w_dict["encoderblock_{}/MultiHeadAttention/out/kernel".format(i)] = out_kernel

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
    model = vit_base_patch16_224_in21k()
    model.build((1, 224, 224, 3))
    # https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
    main(weights_path="./ViT-B_16.npz",
         model_name="ViT-B_16",
         model=model)

    # model = vit_base_patch32_224_in21k()
    # model.build((1, 224, 224, 3))
    # # https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz
    # main(weights_path="./ViT-B_32.npz",
    #      model_name="ViT-B_32",
    #      model=model)

    # model = vit_large_patch16_224_in21k()
    # model.build((1, 224, 224, 3))
    # # https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz
    # main(weights_path="./ViT-L_16.npz",
    #      model_name="ViT-L_16",
    #      model=model)

    # model = vit_large_patch32_224_in21k()
    # model.build((1, 224, 224, 3))
    # # https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz
    # main(weights_path="./ViT-L_32.npz",
    #      model_name="ViT-L_32",
    #      model=model)
