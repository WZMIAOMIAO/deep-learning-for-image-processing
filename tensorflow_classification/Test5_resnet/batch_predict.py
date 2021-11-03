import os
import json
import glob

import tensorflow as tf
import numpy as np
from PIL import Image

from model import resnet50


def main():
    im_height = 224
    im_width = 224
    num_classes = 5

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94

    # load images
    # 指向需要遍历预测的图像文件夹
    imgs_root = "/data/imgs"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    feature = resnet50(num_classes=num_classes, include_top=False)
    feature.trainable = False
    model = tf.keras.Sequential([feature,
                                 tf.keras.layers.GlobalAvgPool2D(),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(1024, activation="relu"),
                                 tf.keras.layers.Dropout(rate=0.5),
                                 tf.keras.layers.Dense(num_classes),
                                 tf.keras.layers.Softmax()])

    # load weights
    weights_path = './save_weights/resNet_50.ckpt'
    assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)

    batch_size = 8  # 每次预测时将多少张图片打包成一个batch
    for ids in range(0, len(img_path_list) // batch_size):
        img_list = []
        for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
            assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
            img = Image.open(img_path)
            # resize image to 224x224
            img = img.resize((im_width, im_height))

            # scaling pixel value to (0-1)
            img = np.array(img).astype(np.float32)
            img = img - [_R_MEAN, _G_MEAN, _B_MEAN]
            img_list.append(img)

        # batch images
        # 将img_list列表中的所有图像打包成一个batch
        batch_img = np.stack(img_list, axis=0)

        # prediction
        result = model.predict(batch_img)
        predict_classes = np.argmax(result, axis=1)

        for index, class_index in enumerate(predict_classes):
            print_res = "image: {}  class: {}   prob: {:.3}".format(img_path_list[ids * batch_size + index],
                                                                    class_indict[str(class_index)],
                                                                    result[index][class_index])
            print(print_res)


if __name__ == '__main__':
    main()
