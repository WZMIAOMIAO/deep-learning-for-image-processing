import sys
import math
from typing import Callable, List

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .losses import WingLoss
from .metrics import NMEMetric
from .distributed_utils import is_main_process, reduce_value


def train_one_epoch(model: torch.nn.Module,
                    epoch: int,
                    train_loader: DataLoader,
                    device: torch.device,
                    optimizer: torch.optim.Optimizer,
                    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                    tb_writer: SummaryWriter,
                    num_keypoints: int,
                    img_hw: List[int]) -> None:
    # define loss function
    loss_func = WingLoss()
    wh_tensor = torch.as_tensor(img_hw[::-1], dtype=torch.float32, device=device).reshape([1, 1, 2])

    model.train()
    train_bar = train_loader
    if is_main_process():
        train_bar = tqdm(train_loader, file=sys.stdout)

    for step, (imgs, targets) in enumerate(train_bar):
        imgs = imgs.to(device)
        labels = targets["keypoints"].to(device)

        optimizer.zero_grad()
        # use mixed precision to speed up training
        with torch.autocast(device_type=device.type):
            pred: torch.Tensor = model(imgs)
            loss: torch.Tensor = loss_func(pred.reshape((-1, num_keypoints, 2)), labels, wh_tensor)

        loss_value = reduce_value(loss).item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if is_main_process():
            train_bar.desc = f"train epoch[{epoch}] loss:{loss_value:.3f}"

            global_step = epoch * len(train_loader) + step
            tb_writer.add_scalar("train loss", loss.item(), global_step=global_step)
            tb_writer.add_scalar("learning rate", optimizer.param_groups[0]["lr"], global_step=global_step)


@torch.inference_mode()
def evaluate(model: torch.nn.Module,
             epoch: int,
             val_loader: DataLoader,
             device: torch.device,
             tb_writer: SummaryWriter,
             affine_points_torch_func: Callable,
             num_keypoints: int,
             img_hw: List[int]) -> None:
    model.eval()
    metric = NMEMetric(device=device)
    wh_tensor = torch.as_tensor(img_hw[::-1], dtype=torch.float32, device=device).reshape([1, 1, 2])
    eval_bar = val_loader
    if is_main_process():
        eval_bar = tqdm(val_loader, file=sys.stdout, desc="evaluation")

    for step, (imgs, targets) in enumerate(eval_bar):
        imgs = imgs.to(device)
        m_invs = targets["m_invs"].to(device)
        labels = targets["ori_keypoints"].to(device)

        pred = model(imgs)
        pred = pred.reshape((-1, num_keypoints, 2))  # [N, K, 2]
        pred = pred * wh_tensor  # rel coord to abs coord
        pred = affine_points_torch_func(pred, m_invs)

        metric.update(pred, labels)

    metric.synchronize_results()
    if is_main_process():
        nme = metric.evaluate()
        tb_writer.add_scalar("evaluation nme", nme, global_step=epoch)
        print(f"evaluation NME[{epoch}]: {nme:.3f}")
