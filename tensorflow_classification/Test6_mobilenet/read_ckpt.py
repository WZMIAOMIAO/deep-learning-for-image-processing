"""
建议直接下载使用我转好的权重
链接: https://pan.baidu.com/s/1YgFoIKHqooMrTQg_IqI2hA  密码: 2qht
"""
import tensorflow as tf


def rename_var(ckpt_path, new_ckpt_path, num_classes, except_list):
    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as sess:
        var_list = tf.train.list_variables(ckpt_path)
        new_var_list = []

        for var_name, shape in var_list:
            # print(var_name)
            if var_name in except_list:
                continue
            if "RMSProp" in var_name or "Exponential" in var_name:
                continue
            var = tf.train.load_variable(ckpt_path, var_name)
            new_var_name = var_name.replace('MobilenetV2/', "")
            new_var_name = new_var_name.replace("/expand/weights", "/expand/Conv2d/weights")
            new_var_name = new_var_name.replace("Conv/weights", "Conv/Conv2d/kernel")
            new_var_name = new_var_name.replace("Conv_1/weights", "Conv_1/Conv2d/kernel")
            new_var_name = new_var_name.replace("weights", "kernel")
            new_var_name = new_var_name.replace("biases", "bias")

            first_word = new_var_name.split('/')[0]
            if "expanded_conv" in first_word:
                last_word = first_word.split('expanded_conv')[-1]
                if len(last_word) > 0:
                    new_word = "inverted_residual" + last_word + "/expanded_conv/"
                else:
                    new_word = "inverted_residual/expanded_conv/"
                new_var_name = new_word + new_var_name.split('/', maxsplit=1)[-1]
            print(new_var_name)
            re_var = tf.Variable(var, name=new_var_name)
            new_var_list.append(re_var)

        re_var = tf.Variable(tf.keras.initializers.he_uniform()([1280, num_classes]), name="Logits/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([num_classes]), name="Logits/bias")

        new_var_list.append(re_var)
        tf.keras.initializers.he_uniform()
        saver = tf.compat.v1.train.Saver(new_var_list)
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, save_path=new_ckpt_path, write_meta_graph=False, write_state=False)


def main():
    except_list = ['global_step', 'MobilenetV2/Logits/Conv2d_1c_1x1/biases', 'MobilenetV2/Logits/Conv2d_1c_1x1/weights']
    ckpt_path = './pretrain_model/mobilenet_v2_1.0_224.ckpt'
    new_ckpt_path = './pretrain_weights.ckpt'
    num_classes = 5
    rename_var(ckpt_path, new_ckpt_path, num_classes, except_list)


if __name__ == '__main__':
    main()
