import os
import json
import time

import torch
from PIL import Image
import matplotlib.pyplot as plt

import transforms
from src import SSD300, Backbone
from draw_box_utils import draw_objs


def create_model(num_classes):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # create model
    # 目标检测数 + 背景
    num_classes = 20 + 1
    model = create_model(num_classes=num_classes)

    # load train weights
    train_weights = "./save_weights/ssd300-14.pth"
    model.load_state_dict(torch.load(train_weights, map_location='cpu')['model'])
    model.to(device)

    # read class_indict
    json_path = "./pascal_voc_classes.json"
    assert os.path.exists(json_path), "file '{}' dose not exist.".format(json_path)
    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    original_img = Image.open("./test.jpg")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.Resize(),
                                         transforms.ToTensor(),
                                         transforms.Normalization()])
    img, _ = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        # initial model
        init_img = torch.zeros((1, 3, 300, 300), device=device)
        model(init_img)

        time_start = time_synchronized()
        predictions = model(img.to(device))[0]  # bboxes_out, labels_out, scores_out
        time_end = time_synchronized()
        print("inference+NMS time: {}".format(time_end - time_start))

        predict_boxes = predictions[0].to("cpu").numpy()
        predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * original_img.size[0]
        predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * original_img.size[1]
        predict_classes = predictions[1].to("cpu").numpy()
        predict_scores = predictions[2].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        plot_img.save("test_result.jpg")


if __name__ == "__main__":
    main()
