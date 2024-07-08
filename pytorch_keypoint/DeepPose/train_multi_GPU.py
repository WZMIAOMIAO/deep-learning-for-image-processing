import os

import torch
import torch.amp
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter

import transforms
from model import create_deep_pose_model
from datasets import WFLWDataset
from train_utils.train_eval_utils import train_one_epoch, evaluate
from train_utils.distributed_utils import init_distributed_mode, is_main_process


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch DeepPose Training", add_help=add_help)
    parser.add_argument("--dataset_dir", type=str, default="/home/wz/datasets/WFLW", help="WFLW dataset directory")
    parser.add_argument("--device", type=str, default="cuda", help="training device, e.g. cpu, cuda")
    parser.add_argument("--save_weights_dir", type=str, default="./weights", help="save dir for model weights")
    parser.add_argument("--save_freq", type=int, default=5, help="save frequency for weights and generated imgs")
    parser.add_argument("--eval_freq", type=int, default=5, help="evaluate frequency")
    parser.add_argument('--img_hw', default=[256, 256], nargs='+', type=int, help='training image size[h, w]')
    parser.add_argument("--epochs", type=int, default=210, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers, default: 8")
    parser.add_argument("--num_keypoints", type=int, default=98, help="number of keypoints")
    parser.add_argument("--lr", type=float, default=5e-4, help="Adam: learning rate")
    parser.add_argument('--lr_steps', default=[170, 200], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument("--warmup_epoch", type=int, default=10, help="number of warmup epoch for training")
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--test_only', action="store_true", help='Only test the model')

    return parser


def main(args):
    torch.manual_seed(1234)
    init_distributed_mode(args)
    if not args.distributed:
        raise EnvironmentError("not support distributed training.")

    dataset_dir = args.dataset_dir
    save_weights_dir = args.save_weights_dir
    save_freq = args.save_freq
    eval_freq = args.eval_freq
    num_keypoints = args.num_keypoints
    num_workers = args.num_workers
    epochs = args.epochs
    bs = args.batch_size
    start_epoch = 0
    img_hw = args.img_hw
    device = torch.device(args.device)
    os.makedirs(save_weights_dir, exist_ok=True)

    # adjust learning rate
    args.lr = args.lr * args.world_size

    tb_writer = None
    if is_main_process():
        # tensorboard writer
        tb_writer = SummaryWriter()

    # create model
    model = create_deep_pose_model(num_keypoints)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # config dataset and dataloader
    data_transform = {
        "train": transforms.Compose([
            transforms.AffineTransform(scale_factor=(0.65, 1.35), rotate=45, shift_factor=0.15, fixed_size=img_hw),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(scale_prob=0., rotate_prob=0., shift_prob=0., fixed_size=img_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    train_dataset = WFLWDataset(root=dataset_dir,
                                train=True,
                                transforms=data_transform["train"])
    val_dataset = WFLWDataset(root=dataset_dir,
                              train=False,
                              transforms=data_transform["val"])

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_loader = DataLoader(train_dataset,
                              batch_sampler=train_batch_sampler,
                              pin_memory=True,
                              num_workers=num_workers,
                              collate_fn=WFLWDataset.collate_fn,
                              persistent_workers=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=bs,
                            sampler=val_sampler,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers,
                            collate_fn=WFLWDataset.collate_fn,
                            persistent_workers=True)

    # define optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # define learning rate scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=len(train_loader) * args.warmup_epoch
    )
    multi_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[len(train_loader) * i for i in args.lr_steps],
        gamma=0.1
    )

    lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, multi_step_scheduler])

    if args.resume:
        assert os.path.exists(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(start_epoch))

    if args.test_only:
        evaluate(model=model,
                 epoch=start_epoch,
                 val_loader=val_loader,
                 device=device,
                 tb_writer=tb_writer,
                 affine_points_torch_func=transforms.affine_points_torch,
                 num_keypoints=num_keypoints,
                 img_hw=img_hw)
        return

    for epoch in range(start_epoch, epochs):
        # train
        train_sampler.set_epoch(epoch)  # shuffle training data
        train_one_epoch(model=model,
                        epoch=epoch,
                        train_loader=train_loader,
                        device=device,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        tb_writer=tb_writer,
                        num_keypoints=num_keypoints,
                        img_hw=img_hw)

        # eval
        if epoch % eval_freq == 0 or epoch == args.epochs - 1:
            evaluate(model=model,
                     epoch=epoch,
                     val_loader=val_loader,
                     device=device,
                     tb_writer=tb_writer,
                     affine_points_torch_func=transforms.affine_points_torch,
                     num_keypoints=num_keypoints,
                     img_hw=img_hw)

        # save weights
        if is_main_process() and (epoch % save_freq == 0 or epoch == args.epochs - 1):
            save_files = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch
            }
            torch.save(save_files, os.path.join(save_weights_dir, f"model_weights_{epoch}.pth"))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
