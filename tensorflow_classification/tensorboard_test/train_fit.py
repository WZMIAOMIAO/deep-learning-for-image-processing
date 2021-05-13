import json
import os
import math
import datetime

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    batch_size = 8
    epochs = 20
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.CategoricalAccuracy("accuracy")])

    # 自定义学习率变化
    def scheduler(epoch):
        initial_lr = 0.01
        end_lr = 0.001
        rate = ((1 + math.cos(epoch * math.pi / epochs)) / 2) * (1 - end_lr) + end_lr  # cosine
        new_lr = rate * initial_lr

        return new_lr

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/model_{epoch}.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor=tf.keras.metrics.CategoricalAccuracy("accuracy").name),
                 tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                write_graph=True,
                                                histogram_freq=1),
                 tf.keras.callbacks.LearningRateScheduler(schedule=scheduler)]

    model.fit(x=train_data_gen,
              epochs=epochs,
              validation_data=val_data_gen,
              callbacks=callbacks)


if __name__ == '__main__':
    main()
