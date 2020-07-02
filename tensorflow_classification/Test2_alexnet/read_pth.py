import torch
import numpy as np
import tensorflow as tf


def rename_var(pth_path, new_ckpt_path, num_classes):
    pytorch_dict = torch.load(pth_path)

    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as sess:
        new_var_list = []

        for key, value in pytorch_dict.items():
            if key in except_list:
                continue

            new_name = key
            value = value.detach().numpy()

            if 'features.0' in new_name:
                new_name = new_name.replace("features.0.weight", "conv2d/kernel")
                new_name = new_name.replace("features.0.bias", "conv2d/bias")

            if 'features.3' in new_name:
                new_name = new_name.replace("features.3.weight", "conv2d_1/kernel")
                new_name = new_name.replace("features.3.bias", "conv2d_1/bias")

            if 'features.6' in new_name:
                new_name = new_name.replace("features.6.weight", "conv2d_2/kernel")
                new_name = new_name.replace("features.6.bias", "conv2d_2/bias")

            if 'features.8' in new_name:
                new_name = new_name.replace("features.8.weight", "conv2d_3/kernel")
                new_name = new_name.replace("features.8.bias", "conv2d_3/bias")

            if 'features.10' in new_name:
                new_name = new_name.replace("features.10.weight", "conv2d_4/kernel")
                new_name = new_name.replace("features.10.bias", "conv2d_4/bias")

            if 'classifier.1' in new_name:
                new_name = new_name.replace("classifier.1.weight", "dense/kernel")
                new_name = new_name.replace("classifier.1.bias", "dense/bias")

            if 'classifier.4' in new_name:
                new_name = new_name.replace("classifier.4.weight", "dense_1/kernel")
                new_name = new_name.replace("classifier.4.bias", "dense_1/bias")

            if 'conv2d' in new_name and 'kernel' in new_name:
                value = np.transpose(value, (2, 3, 1, 0)).astype(np.float32)
            else:
                value = np.transpose(value).astype(np.float32)

            re_var = tf.Variable(value, name=new_name)
            new_var_list.append(re_var)

        re_var = tf.Variable(tf.keras.initializers.he_uniform()([4096, num_classes]), name="dense_2/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([num_classes]), name="dense_2/bias")
        new_var_list.append(re_var)

        saver = tf.compat.v1.train.Saver(new_var_list)
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, save_path=new_ckpt_path, write_meta_graph=False, write_state=False)


except_list = ['classifier.6.weight', 'classifier.6.bias']
# https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
pth_path = './alexnet-owt-4df8aa71.pth'
new_ckpt_path = './pretrain_weights.ckpt'
num_classes = 5
rename_var(pth_path, new_ckpt_path, num_classes)