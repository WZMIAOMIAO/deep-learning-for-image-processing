import os
import torch
from torchvision import transforms, datasets
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from tqdm import tqdm
import time
from model import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path
batch_size = 16


def validate_model(model: torch.nn.Module):
    validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                            transform=data_transform)
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)

    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        t1 = time.time()
        for val_data in tqdm(validate_loader, desc="validate model accuracy."):
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))  # eval model only have last output layer
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.sum(torch.eq(predict_y, val_labels.to(device))).item()
        val_accurate = acc / val_num
        print('test_accuracy: %.3f, time:%.3f' % (val_accurate, time.time() - t1))

    return val_accurate


def count_sparsity(model: torch.nn.Module, p=True):
    sum_zeros_num = 0
    sum_weights_num = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            zeros_elements = torch.sum(torch.eq(module.weight, 0)).item()
            weights_elements = module.weight.numel()

            sum_zeros_num += zeros_elements
            sum_weights_num += weights_elements
            if p is True:
                print("Sparsity in {}.weights {:.2f}%".format(name, 100 * zeros_elements / weights_elements))
    print("Global sparsity: {:.2f}%".format(100 * sum_zeros_num / sum_weights_num))


def main():
    weights_path = "./resNet34.pth"
    model = resnet34(num_classes=5)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    # validate_model(model)
    # module = model.conv1
    # print(list(module.named_parameters()))
    # # print(list(module.named_buffers()))
    #
    # # 裁剪50%的卷积核
    # prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
    # print(list(module.weight))
    # print(module.weight.shape)
    # # print(list(module.named_buffers()))
    #
    # prune.remove(module, "weight")
    # print(module.weight.shape)

    # 收集所有需要裁剪的卷积核
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, "weight"))

    # 对卷积核进行剪枝处理
    prune.global_unstructured(parameters_to_prune,
                              pruning_method=prune.L1Unstructured,
                              amount=0.5)

    # 统计剪枝比例
    count_sparsity(model, p=False)

    # 验证剪枝后的模型
    validate_model(model)
    # print(model)

    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         prune.remove(module, "weight")
    # validate_model(model)

    # torch.save(model.state_dict(), "pruning_model.pth")


if __name__ == '__main__':
    main()
