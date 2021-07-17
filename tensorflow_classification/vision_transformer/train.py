import os
import re
import math
import datetime

import tensorflow as tf
from tqdm import tqdm

from vit_model import vit_base_patch16_224_in21k as create_model
from utils import generate_ds

assert tf.version.VERSION >= "2.4.0", "version of tf must greater/equal than 2.4.0"


def main():
    data_root = "/data/flower_photos"  # get data root path

    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")

    batch_size = 8
    epochs = 10
    num_classes = 5
    freeze_layers = True
    initial_lr = 0.001
    weight_decay = 1e-4

    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    # data generator with data augmentation
    train_ds, val_ds = generate_ds(data_root, batch_size=batch_size, val_rate=0.2)

    # create model
    model = create_model(num_classes=num_classes, has_logits=False)
    model.build((1, 224, 224, 3))

    # 下载我提前转好的预训练权重
    # 链接: https://pan.baidu.com/s/1ro-6bebc8zroYfupn-7jVQ  密码: s9d9
    # load weights
    pre_weights_path = './ViT-B_16.h5'
    assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    # freeze bottom layers
    if freeze_layers:
        for layer in model.layers:
            if "pre_logits" not in layer.name and "head" not in layer.name:
                layer.trainable = False
            else:
                print("training {}".format(layer.name))

    model.summary()

    # custom learning rate curve
    def scheduler(now_epoch):
        end_lr_rate = 0.01  # end_lr = initial_lr * end_lr_rate
        rate = ((1 + math.cos(now_epoch * math.pi / epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
        new_lr = rate * initial_lr

        # writing lr into tensorboard
        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)

        return new_lr

    # using keras low level api for training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            output = model(train_images, training=True)
            # cross entropy loss
            ce_loss = loss_object(train_labels, output)

            # l2 loss
            matcher = re.compile(".*(bias|gamma|beta).*")
            l2loss = weight_decay * tf.add_n([
                tf.nn.l2_loss(v)
                for v in model.trainable_variables
                if not matcher.match(v.name)
            ])

            loss = ce_loss + l2loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(ce_loss)
        train_accuracy(train_labels, output)

    @tf.function
    def val_step(val_images, val_labels):
        output = model(val_images, training=False)
        loss = loss_object(val_labels, output)

        val_loss(loss)
        val_accuracy(val_labels, output)

    best_val_acc = 0.
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        # train
        train_bar = tqdm(train_ds)
        for images, labels in train_bar:
            train_step(images, labels)

            # print train process
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 train_loss.result(),
                                                                                 train_accuracy.result())

        # update learning rate
        optimizer.learning_rate = scheduler(epoch)

        # validate
        val_bar = tqdm(val_ds)
        for images, labels in val_bar:
            val_step(images, labels)

            # print val process
            val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                               epochs,
                                                                               val_loss.result(),
                                                                               val_accuracy.result())
        # writing training loss and acc
        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        # writing validation loss and acc
        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)

        # only save best weights
        if val_accuracy.result() > best_val_acc:
            best_val_acc = val_accuracy.result()
            save_name = "./save_weights/model.ckpt"
            model.save_weights(save_name, save_format="tf")


if __name__ == '__main__':
    main()
