import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from model import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    weights_path = "./resNet34.pth"
    model = resnet34(num_classes=5)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    module = model.conv1
    print(list(module.named_parameters()))
    # print(list(module.named_buffers()))

    # 裁剪50%的卷积核
    prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
    print(list(module.weight))
    print(module.weight.shape)
    # print(list(module.named_buffers()))

    prune.remove(module, "weight")
    print(module.weight.shape)


if __name__ == '__main__':
    main()
