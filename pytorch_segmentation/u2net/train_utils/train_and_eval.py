import torch
from torch.nn import functional as F
import train_utils.distributed_utils as utils


def criterion(inputs, target):
    losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
    total_loss = sum(losses)

    return total_loss


def evaluate(model, data_loader, device):
    model.eval()
    mae_metric = utils.MeanAbsoluteError()
    f1_metric = utils.F1Score()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images, targets = images.to(device), targets.to(device)
            output = model(images)

            mae_metric.update(output, targets)
            f1_metric.update(output, targets)

        mae_metric.gather_from_all_processes()
        f1_metric.reduce_from_all_processes()

    return mae_metric, f1_metric


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr
