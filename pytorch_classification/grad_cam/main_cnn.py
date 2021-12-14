import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image


def main():
    model = models.mobilenet_v3_large(pretrained=True)
    # model = models.vgg16(pretrained=True)
    # model = models.resnet34(pretrained=True)
    # model = models.regnet_y_800mf(pretrained=True)
    # model = models.efficientnet_b0(pretrained=True)
    target_layers = [model.features[-1]]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "1.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    rgb_img = np.array(img, dtype=np.float32) / 255.
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    input_tensor = torch.unsqueeze(img, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 281  # cat

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
