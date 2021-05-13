"""
可直接下载我转好的权重
链接: https://pan.baidu.com/s/1tLe9ahTMIwQAX7do_S59Zg  密码: u199
"""
import tensorflow as tf


def rename_var(ckpt_path, new_ckpt_path, num_classes, except_list):
    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as sess:
        var_list = tf.train.list_variables(ckpt_path)
        new_var_list = []

        for var_name, shape in var_list:
            print(var_name)
            if var_name in except_list:
                continue
            var = tf.train.load_variable(ckpt_path, var_name)
            new_var_name = var_name.replace('resnet_v1_50/', "")
            new_var_name = new_var_name.replace("bottleneck_v1/", "")
            new_var_name = new_var_name.replace("shortcut/weights", "shortcut/conv1/kernel")
            new_var_name = new_var_name.replace("weights", "kernel")
            new_var_name = new_var_name.replace("biases", "bias")
            re_var = tf.Variable(var, name=new_var_name)
            new_var_list.append(re_var)

        re_var = tf.Variable(tf.keras.initializers.he_uniform()([2048, num_classes]), name="logits/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([num_classes]), name="logits/bias")
        new_var_list.append(re_var)
        saver = tf.compat.v1.train.Saver(new_var_list)
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, save_path=new_ckpt_path, write_meta_graph=False, write_state=False)


def main():
    except_list = ['global_step', 'resnet_v1_50/mean_rgb', 'resnet_v1_50/logits/biases', 'resnet_v1_50/logits/weights']
    ckpt_path = './resnet_v1_50.ckpt'
    new_ckpt_path = './pretrain_weights.ckpt'
    num_classes = 5
    rename_var(ckpt_path, new_ckpt_path, num_classes, except_list)


if __name__ == '__main__':
    main()
