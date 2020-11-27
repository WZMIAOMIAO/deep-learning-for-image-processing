from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import vgg
import tensorflow as tf
import json
import os
import glob


def main():
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # create direction for saving weights
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    def pre_function(img):
        # img = im.open('test.jpg')
        # img = np.array(img).astype(np.float32)
        img = img - [_R_MEAN, _G_MEAN, _B_MEAN]

        return img

    # data generator with data augmentation
    train_image_generator = ImageDataGenerator(horizontal_flip=True,
                                               preprocessing_function=pre_function)
    validation_image_generator = ImageDataGenerator(preprocessing_function=pre_function)

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

    model = vgg("vgg16", 224, 224, 5)

    pre_weights_path = './pretrain_weights.ckpt'
    assert len(glob.glob(pre_weights_path+"*")), "cannot find {}".format(pre_weights_path)
    model.load_weights(pre_weights_path)
    for layer_t in model.layers:
        if layer_t.name == 'feature':
            layer_t.trainable = False
            break

    model.summary()

    # using keras high level api for training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex_{epoch}.h5',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss')]

    # tensorflow2.1 recommend to using fit
    history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks)


if __name__ == '__main__':
    main()
