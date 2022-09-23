import torch
from alexnet_model import AlexNet
from resnet_model import resnet34
# from resnet_model import  resnet50 as creat_model
from swin_model import  swin_tiny_patch4_window7_224 as creat_model_swin
from resnet_model import resnet50 as creat_model_res50
from alexnet_model import AlexNet as creat_model_alex
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# data_transform = transforms.Compose(
#     [transforms.Resize((224, 224)),
#      transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                     # transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# data_transform = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(224),
#      transforms.ToTensor(),
#      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create model
model = creat_model_res50(num_classes=6)
# model = resnet34(num_classes=5)

# load model weights
model_weight_path = "./resnet50_adamw_lr0.0001_wd5e-2_lrf0.01.pth"  # "./resNet34.pth"
model.load_state_dict(torch.load(model_weight_path))
# print(model)

# print('-----------------------------------------------------')

# load image
img = Image.open("./10000_00000032.jpg")
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# forward
out_put = model(img)
# print(out_put)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    # print(feature_map.shape)
    print(feature_map.shape)
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C] 对普通卷积
    im = np.transpose(im, [1, 2, 0])
    # print('im.shape:',im.shape)

    # show top 12 feature maps
    plt.figure()
    for i in range(48):
        ax = plt.subplot(6, 8, i+1)
        # [H, W, C]
        plt.axis('off')
        plt.imshow(im[:, :, i],cmap='gray')
    plt.show()

