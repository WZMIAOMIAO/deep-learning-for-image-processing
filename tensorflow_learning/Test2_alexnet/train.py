from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v1, AlexNet_v2
import tensorflow as tf
import json
import os

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path
train_dir = image_path + "train"
validation_dir = image_path + "val"

im_height = 224
im_width = 224
batch_size = 32
epochs = 2

# data generator with data augmentation
train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                           horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

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
                                                              shuffle=True,
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')
total_val = val_data_gen.n

# sample_training_images, sample_training_labels = next(train_data_gen)


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(20, 20))
#     axes = axes.flatten()
#     for img, ax in zip(images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()


# plotImages(sample_training_images[:5])

model = AlexNet_v1(im_height=im_height, im_width=im_width, class_num=5)
# model = AlexNet_v2(class_num=5)
# model.build((batch_size, 224, 224, 3))  # when using subclass model
model.summary()

# using keras high level api for training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
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

# history = model.fit_generator(generator=train_data_gen,
#                               steps_per_epoch=total_train // batch_size,
#                               epochs=epochs,
#                               validation_data=val_data_gen,
#                               validation_steps=total_val // batch_size,
#                               callbacks=callbacks)

# # using keras low level api for training
# loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
#
# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
#
# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
#
#
# @tf.function
# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         predictions = model(images)
#         loss = loss_object(labels, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(labels, predictions)
#
#
# @tf.function
# def test_step(images, labels):
#     predictions = model(images)
#     t_loss = loss_object(labels, predictions)
#
#     test_loss(t_loss)
#     test_accuracy(labels, predictions)
#
#
# best_test_loss = float('inf')
# for epoch in range(1, epochs+1):
#     train_loss.reset_states()        # clear history info
#     train_accuracy.reset_states()    # clear history info
#     test_loss.reset_states()         # clear history info
#     test_accuracy.reset_states()     # clear history info
#     for step in range(total_train // batch_size):
#         images, labels = next(train_data_gen)
#         train_step(images, labels)
#
#     for step in range(total_val // batch_size):
#         test_images, test_labels = next(val_data_gen)
#         test_step(test_images, test_labels)
#
#     template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
#     print(template.format(epoch,
#                           train_loss.result(),
#                           train_accuracy.result() * 100,
#                           test_loss.result(),
#                           test_accuracy.result() * 100))
#     if test_loss.result() < best_test_loss:
#         model.save_weights("./save_weights/myAlex_{}.ckpt".format(epoch), save_format='tf')
