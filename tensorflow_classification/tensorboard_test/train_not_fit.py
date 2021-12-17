import json
import os
import math
import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
assert tf.version.VERSION >= "2.4.0", "version of tf must greater/equal than 2.4.0"


def main():
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights")

    num_classes = 5
    im_height = 224
    im_width = 224
    batch_size = 16
    epochs = 20
    log_dir = "./logs/not_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

    # data generator with data augmentation
    train_image_generator = ImageDataGenerator(horizontal_flip=True)

    validation_image_generator = ImageDataGenerator()

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')
    total_train = train_data_gen.n

    # get class dict
    class_indices = train_data_gen.class_indices

    # transform value and key of dict
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')

    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))

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

    # 自定义学习率变化
    def scheduler(epoch):
        initial_lr = 0.01
        end_lr = 0.001
        rate = ((1 + math.cos(epoch * math.pi / epochs)) / 2) * (1 - end_lr) + end_lr  # cosine
        new_lr = rate * initial_lr

        with train_writer.as_default():
            tf.summary.scalar('learning rate', data=new_lr, step=epoch)

        return new_lr

    # using keras low level api for training
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            output = model(images, training=True)
            loss = loss_object(labels, output)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, output)

    @tf.function
    def test_step(images, labels):
        output = model(images, training=False)
        t_loss = loss_object(labels, output)

        val_loss(t_loss)
        val_accuracy(labels, output)

    best_val_accuracy = 0.
    for epoch in range(epochs):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        val_loss.reset_states()  # clear history info
        val_accuracy.reset_states()  # clear history info

        print("Epoch [{}/{}]".format(epoch + 1, epochs))
        # train
        train_bar = tqdm(train_data_gen, file=sys.stdout)
        for images, labels in train_bar:
            train_step(images, labels)

            # print train process
            train_bar.desc = "train_loss:{:.3f}, train_acc:{:.3f}".format(train_loss.result(),
                                                                          train_accuracy.result())

        # update learning rate
        optimizer.learning_rate = scheduler(epoch)

        # validation
        val_bar = tqdm(val_data_gen, file=sys.stdout)
        for test_images, test_labels in val_bar:
            test_step(test_images, test_labels)

            # print val process
            val_bar.desc = "val_loss:{:.3f}, val_acc:{:.3f}".format(val_loss.result(),
                                                                    val_accuracy.result())

        with train_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), epoch)

        with val_writer.as_default():
            tf.summary.scalar("loss", val_loss.result(), epoch)
            tf.summary.scalar("accuracy", val_accuracy.result(), epoch)

        if val_accuracy.result() > best_val_accuracy:
            best_val_accuracy = val_accuracy.result()
            model.save_weights("./save_weights/model_{}.ckpt".format(epoch), save_format="tf")


if __name__ == '__main__':
    main()
