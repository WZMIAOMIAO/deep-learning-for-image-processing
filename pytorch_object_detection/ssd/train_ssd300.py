from src.ssd_model import SSD300, Backbone
import torch
import transform
from my_dataset import VOC2012DataSet
import os
import train_utils.train_eval_utils as utils
from train_utils.coco_utils import get_coco_api_from_dataset


def create_model(num_classes=21):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    pre_train_path = "./src/resnet50.pth"
    backbone = Backbone(pretrain_path=pre_train_path)
    model = SSD300(backbone=backbone, num_classes=num_classes)

    pre_ssd_path = "./src/nvidia_ssdpyt_fp32.pt"
    pre_model_dict = torch.load(pre_ssd_path, map_location=torch.device('cpu'))
    pre_weights_dict = pre_model_dict["model"]

    # 删除类别预测器权重，注意，回归预测器的权重可以重用，因为不涉及num_classes
    del_conf_loc_dict = {}
    for k, v in pre_weights_dict.items():
        split_key = k.split(".")
        if "conf" in split_key:
            continue
        del_conf_loc_dict.update({k: v})

    missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    data_transform = {
        "train": transform.Compose([transform.SSDCropping(),
                                    transform.Resize(),
                                    transform.ColorJitter(),
                                    transform.ToTensor(),
                                    transform.RandomHorizontalFlip(),
                                    transform.Normalization(),
                                    transform.AssignGTtoDefaultBox()]),
        "val": transform.Compose([transform.Resize(),
                                  transform.ToTensor(),
                                  transform.Normalization()])
    }

    voc_path = "../"
    train_dataset = VOC2012DataSet(voc_path, data_transform['train'], True)
    # 注意训练时，batch_size必须大于1
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=8,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    collate_fn=utils.collate_fn)

    val_dataset = VOC2012DataSet(voc_path, data_transform['val'], False)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  collate_fn=utils.collate_fn)

    model = create_model(num_classes=21)
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.3)

    train_loss = []
    learning_rate = []
    val_map = []

    val_data = None
    # 如果电脑内存充裕，可提前加载验证集数据，以免每次验证时都要重新加载一次数据，节省时间
    # val_data = get_coco_api_from_dataset(val_data_loader.dataset)
    for epoch in range(20):
        utils.train_one_epoch(model=model, optimizer=optimizer,
                              data_loader=train_data_loader,
                              device=device, epoch=epoch,
                              print_freq=50, train_loss=train_loss,
                              train_lr=learning_rate, warmup=True)

        lr_scheduler.step()

        utils.evaluate(model=model, data_loader=val_data_loader,
                       device=device, data_set=val_data, mAP_list=val_map)

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./save_weights/ssd300-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)

    # inputs = torch.rand(size=(2, 3, 300, 300))
    # output = model(inputs)
    # print(output)


if __name__ == '__main__':
    main()
