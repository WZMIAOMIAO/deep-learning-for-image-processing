import tensorflow as tf


def rename_var(ckpt_path, new_ckpt_path):
    with tf.Graph().as_default(), tf.compat.v1.Session().as_default() as sess:
        var_list = tf.train.list_variables(ckpt_path)
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

        re_var = tf.Variable(tf.keras.initializers.he_uniform()([2048, 5]), name="logits/kernel")
        new_var_list.append(re_var)
        re_var = tf.Variable(tf.keras.initializers.he_uniform()([5]), name="logits/bias")
        new_var_list.append(re_var)
        saver = tf.compat.v1.train.Saver(new_var_list)
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, save_path=new_ckpt_path, write_meta_graph=False, write_state=False)


except_list = ['global_step', 'resnet_v1_50/mean_rgb', 'resnet_v1_50/logits/biases', 'resnet_v1_50/logits/weights']
ckpt_path = './resnet_v1_50.ckpt'
new_ckpt_path = './pretrain_weights.ckpt'
new_var_list = []
rename_var(ckpt_path, new_ckpt_path)
