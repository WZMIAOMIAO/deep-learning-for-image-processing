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

            new_name = new_name.replace(".", "/")

            # 将卷积核的通道顺序由pytorch调整到tensorflow
            if 'conv/weight' in new_name:
                new_name = new_name.replace("weight", "kernel")
                value = np.transpose(value, (2, 3, 1, 0)).astype(np.float32)
            elif 'bn' in new_name:
                if "num_batches_tracked" in new_name:
                    continue

                new_name = new_name.replace("weight", "gamma")
                new_name = new_name.replace("bias", "beta")
                new_name = new_name.replace("running_mean", "moving_mean")
                new_name = new_name.replace("running_var", "moving_variance")

                value = np.transpose(value).astype(np.float32)
            elif 'fc1' in new_name:
                new_name = new_name.replace("weight", "kernel")
                value = np.transpose(value).astype(np.float32)

            re_var = tf.Variable(value, name=new_name)
            new_var_list.append(re_var)

        # aux1
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([1024, num_classes]), name="aux1/fc2/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([num_classes]), name="aux1/fc2/bias")
        new_var_list.append(re_var)

        # aux2
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([1024, num_classes]), name="aux2/fc2/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([num_classes]), name="aux2/fc2/bias")
        new_var_list.append(re_var)

        # fc
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([1024, num_classes]), name="fc/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([num_classes]), name="fc/bias")
        new_var_list.append(re_var)

        saver = tf.compat.v1.train.Saver(new_var_list)
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, save_path=new_ckpt_path, write_meta_graph=False, write_state=False)


# this script only use for model_add_bn.py
except_list = ['aux1.fc2.weight', 'aux1.fc2.bias', 'aux2.fc2.weight', 'aux2.fc2.bias', 'fc.weight', 'fc.bias']
# https://download.pytorch.org/models/googlenet-1378be20.pth
pth_path = './googlenet-1378be20.pth'
new_ckpt_path = './pretrain_weights.ckpt'
num_classes = 5
rename_var(pth_path, new_ckpt_path, num_classes)
