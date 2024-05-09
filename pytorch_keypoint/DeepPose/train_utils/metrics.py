import torch

from .distributed_utils import reduce_value, is_dist_avail_and_initialized


class NMEMetric:
    def __init__(self, device: torch.device) -> None:
        # 两眼外角点对应keypoint索引
        self.keypoint_idxs = [60, 72]
        self.nme_accumulator: float = 0.
        self.counter: float = 0.
        self.device = device

    def update(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            pred (shape [N, K, 2]): pred keypoints
            gt (shape [N, K, 2]): gt keypoints
            mask (shape [N, K]): valid keypoints mask
        """

        l2_loss = (pred - gt).pow(2).mean(dim=(1, 2))
        # ion: inter-ocular distance normalized error
        ion = (pred[:, self.keypoint_idxs] - gt[:, self.keypoint_idxs]).pow(2).sum(dim=(1, 2)).pow(0.5)

        valid_ion_mask = ion > 0
        if mask is None:
            mask = valid_ion_mask.unsqueeze(dim=1)
        else:
            mask = torch.logical_and(mask, valid_ion_mask.unsqueeze_(dim=1))
        num_valid = (mask.sum(dim=1) > 0).sum().item()

        # avoid divide by zero
        ion[ion <= 0] = 1e-6

        self.nme_accumulator += l2_loss.div(ion).sum().item()
        self.counter += num_valid

    def evaluate(self):
        return self.nme_accumulator / self.counter

    def synchronize_results(self):
        if is_dist_avail_and_initialized():
            self.nme_accumulator = reduce_value(
                torch.as_tensor(self.nme_accumulator, device=self.device),
                average=False
            ).item()

            self.counter = reduce_value(
                torch.as_tensor(self.counter, device=self.device),
                average=False
            )


if __name__ == '__main__':
    metric = NMEMetric()
    metric.update(pred=torch.randn(32, 98, 2),
                  gt=torch.randn(32, 98, 2),
                  mask=torch.randn(32, 98))
    print(metric.evaluate())
