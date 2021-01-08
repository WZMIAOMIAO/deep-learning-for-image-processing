import os
import math
import datetime

import tensorflow as tf

from utils import generate_ds


def main():
    data_root = "/home/wz/my_project/my_github/data_set/flower_data/flower_photos"  # get data root path

    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")

    num_classes = 5
    im_height = 224
    im_width = 224
    batch_size = 8
    epochs = 20
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_ds, val_ds = generate_ds(data_root, im_height, im_width, batch_size)

    # create base model
    base_model = tf.keras.applications.ResNet50(include_top=False,
                                                input_shape=(224, 224, 3),
                                                weights='imagenet')
    # freeze base model
    base_model.trainable = False
    base_model.summary()

    # create new model on top
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    def scheduler(epoch):
        """
        自定义学习率变化
        :param epoch: 当前训练epoch
        :return:
        """
        initial_lr = 0.01
        end_lr = 0.001
        rate = ((1 + math.cos(epoch * math.pi / epochs)) / 2) * (1 - end_lr) + end_lr  # cosine
        new_lr = rate * initial_lr

        return new_lr

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/model_{epoch}.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_accuracy'),
                 tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                write_graph=True,
                                                histogram_freq=1),
                 tf.keras.callbacks.LearningRateScheduler(schedule=scheduler)]

    model.fit(x=train_ds,
              epochs=epochs,
              validation_data=val_ds,
              callbacks=callbacks)


if __name__ == '__main__':
    main()
