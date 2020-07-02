import tensorflow as tf


def rename_var(ckpt_path, new_ckpt_path, num_classes=5):
    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as sess:
        var_list = tf.train.list_variables(ckpt_path)
        new_var_list = []

        for var_name, shape in var_list:
            # print(var_name)
            if var_name in except_list:
                continue

            var = tf.train.load_variable(ckpt_path, var_name)
            new_var_name = var_name.replace('vgg_16', 'feature')
            new_var_name = new_var_name.replace("weights", "kernel")
            new_var_name = new_var_name.replace("biases", "bias")

            new_var_name = new_var_name.replace("conv1/conv1_1", "conv2d")
            new_var_name = new_var_name.replace("conv1/conv1_2", "conv2d_1")

            new_var_name = new_var_name.replace("conv2/conv2_1", "conv2d_2")
            new_var_name = new_var_name.replace("conv2/conv2_2", "conv2d_3")

            new_var_name = new_var_name.replace("conv3/conv3_1", "conv2d_4")
            new_var_name = new_var_name.replace("conv3/conv3_2", "conv2d_5")
            new_var_name = new_var_name.replace("conv3/conv3_3", "conv2d_6")

            new_var_name = new_var_name.replace("conv4/conv4_1", "conv2d_7")
            new_var_name = new_var_name.replace("conv4/conv4_2", "conv2d_8")
            new_var_name = new_var_name.replace("conv4/conv4_3", "conv2d_9")

            new_var_name = new_var_name.replace("conv5/conv5_1", "conv2d_10")
            new_var_name = new_var_name.replace("conv5/conv5_2", "conv2d_11")
            new_var_name = new_var_name.replace("conv5/conv5_3", "conv2d_12")

            if 'fc' in new_var_name:
                # new_var_name = new_var_name.replace("feature/fc6", "dense")
                # new_var_name = new_var_name.replace("feature/fc7", "dense_1")
                # new_var_name = new_var_name.replace("fc8", "dense_2")
                continue

        #     print(new_var_name)
            re_var = tf.Variable(var, name=new_var_name)
            new_var_list.append(re_var)

        re_var = tf.Variable(tf.keras.initializers.he_uniform()([25088, 2048]), name="dense/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([2048]), name="dense/bias")
        new_var_list.append(re_var)

        re_var = tf.Variable(tf.keras.initializers.he_uniform()([2048, 2048]), name="dense_1/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([2048]), name="dense_1/bias")
        new_var_list.append(re_var)

        re_var = tf.Variable(tf.keras.initializers.he_uniform()([2048, num_classes]), name="dense_2/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([num_classes]), name="dense_2/bias")
        new_var_list.append(re_var)

        saver = tf.compat.v1.train.Saver(new_var_list)
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, save_path=new_ckpt_path, write_meta_graph=False, write_state=False)


except_list = ['global_step', 'vgg_16/mean_rgb', 'vgg_16/fc8/biases', 'vgg_16/fc8/weights']
# http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
ckpt_path = './vgg_16.ckpt'
new_ckpt_path = './pretrain_weights.ckpt'
num_classes = 5
rename_var(ckpt_path, new_ckpt_path, num_classes)
