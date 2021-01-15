import os
import json

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from model import AlexNet_v1, AlexNet_v2


def main():
    im_height = 224
    im_width = 224

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    # resize image to 224x224
    img = img.resize((im_width, im_height))
    plt.imshow(img)

    # scaling pixel value to (0-1)
    img = np.array(img) / 255.

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = AlexNet_v1(num_classes=5)
    weighs_path = "./save_weights/myAlex.h5"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(weighs_path)
    model.load_weights(weighs_path)

    # prediction
    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
