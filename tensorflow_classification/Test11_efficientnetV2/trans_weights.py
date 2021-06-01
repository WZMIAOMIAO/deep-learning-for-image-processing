from model import *


def main(ckpt_path: str,
         model_name: str,
         model: tf.keras.Model):
    var_dict = {v.name.split(':')[0]: v for v in model.weights}

    reader = tf.train.load_checkpoint(ckpt_path)
    var_shape_map = reader.get_variable_to_shape_map()

    for key, var in var_dict.items():
        key_ = model_name + "/" + key
        key_ = key_.replace("batch_normalization", "tpu_batch_normalization")
        if key_ in var_shape_map:
            if var_shape_map[key_] != var.shape:
                msg = "shape mismatch: {}".format(key)
                print(msg)
            else:
                var.assign(reader.get_tensor(key_), read_value=False)
        else:
            msg = "Not found {} in {}".format(key, ckpt_path)
            print(msg)

    model.save_weights("./{}.h5".format(model_name))


if __name__ == '__main__':
    model = efficientnetv2_s()
    model.build((1, 224, 224, 3))
    main(ckpt_path="./efficientnetv2-s-21k-ft1k/model",
         model_name="efficientnetv2-s",
         model=model)

    # model = efficientnetv2_m()
    # model.build((1, 224, 224, 3))
    # main(ckpt_path="./efficientnetv2-m-21k-ft1k/model",
    #      model_name="efficientnetv2-m",
    #      model=model)

    # model = efficientnetv2_l()
    # model.build((1, 224, 224, 3))
    # main(ckpt_path="./efficientnetv2-l-21k-ft1k/model",
    #      model_name="efficientnetv2-l",
    #      model=model)
