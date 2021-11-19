import os
import json
import glob
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from model import efficientnet_b0 as create_model


def main():
    num_classes = 5

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"
    im_height = im_width = img_size[num_model]

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # resize image to 224x224
    img = img.resize((im_width, im_height))
    plt.imshow(img)

    # read image
    img = np.array(img).astype(np.float32)

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=num_classes)

    weights_path = './save_weights/efficientnet.ckpt'
    assert len(glob.glob(weights_path+"*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)

    result = np.squeeze(model.predict(img))
    predict_class = np.argmax(result)

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_class)],
                                                 result[predict_class])
    plt.title(print_res)
    for i in range(len(result)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  result[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
