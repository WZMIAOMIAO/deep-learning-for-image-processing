import os
import json
import glob
import numpy as np

from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from model import swin_tiny_patch4_window7_224 as create_model


def main():
    num_classes = 5
    im_height = im_width = 224

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # resize image
    img = img.resize((im_width, im_height))
    plt.imshow(img)

    # read image
    img = np.array(img).astype(np.float32)

    # preprocess
    img = (img / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=num_classes)
    model.build([1, im_height, im_width, 3])

    weights_path = './save_weights/model.ckpt'
    assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)

    result = np.squeeze(model.predict(img, batch_size=1))
    result = tf.keras.layers.Softmax()(result)
    predict_class = np.argmax(result)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
