import re
import tensorflow as tf
from model_v3 import mobilenet_v3_large


def change_word(word: str):
    word = word.replace("MobilenetV3/", "")

    if "weights" in word:
        word = word.replace("weights", "kernel")
    elif "Conv" in word and "biases" in word:
        word = word.replace("biases", "bias")

    return word


def rename_var(ckpt_path, m_info):
    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as sess:
        var_list = tf.train.list_variables(ckpt_path)
        pattern = "ExponentialMovingAverage|Momentum|global_step"

        var_dict = dict((change_word(name), [name, shape])
                        for name, shape in var_list
                        if len(re.findall(pattern, name)) == 0)

        for k, v in m_info:
            assert k in var_dict, "{} not in var_dict".format(k)
            assert v == var_dict[k][1], "shape {} not equal {}".format(v, var_dict[k][1])

        weights = []
        for k, _ in m_info:
            var = tf.train.load_variable(ckpt_path, var_dict[k][0])
            weights.append(var)

        return weights


def main():
    # https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz
    ckpt_path = './v3-large_224_1.0_float/pristine/model.ckpt-540000'
    save_path = './pre_mobilev3.h5'
    m = mobilenet_v3_large(input_shape=(224, 224, 3), num_classes=1001, include_top=True)
    m_info = [(i.name.replace(":0", ""), list(i.shape))
              for i in m.weights]
    weights = rename_var(ckpt_path, m_info)
    m.set_weights(weights)
    m.save_weights(save_path)


if __name__ == '__main__':
    main()
