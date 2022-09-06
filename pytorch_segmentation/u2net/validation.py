import os
from typing import Union, List

import torch
from torch.utils import data

from src import u2net_full
from train_utils import evaluate
from my_dataset import DUTSDataset
import transforms as T


class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    val_dataset = DUTSDataset(args.data_path, train=False, transforms=SODPresetEval([320, 320]))

    num_workers = 4
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,  # must be 1
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      shuffle=False,
                                      collate_fn=val_dataset.collate_fn)

    model = u2net_full()
    pretrain_weights = torch.load(args.weights, map_location='cpu')
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)

    mae_metric, f1_metric = evaluate(model, val_data_loader, device=device)
    print(mae_metric, f1_metric)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch u2net validation")

    parser.add_argument("--data-path", default="./", help="DUTS root")
    parser.add_argument("--weights", default="./u2net_full.pth")
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
