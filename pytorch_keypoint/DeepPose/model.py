import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def create_deep_pose_model(num_keypoints: int) -> nn.Module:
    res50 = resnet50(ResNet50_Weights.IMAGENET1K_V2)
    in_features = res50.fc.in_features
    res50.fc = nn.Linear(in_features=in_features, out_features=num_keypoints * 2)

    return res50


if __name__ == '__main__':
    torch.manual_seed(1234)
    model = create_deep_pose_model(98)
    model.eval()
    with torch.inference_mode():
        x = torch.randn(1, 3, 224, 224)
        res = model(x)
        print(res.shape)
