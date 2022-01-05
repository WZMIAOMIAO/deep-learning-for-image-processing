import os
import glob
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool):
        super(DriveDataset, self).__init__()
        data_root = os.path.join(root, "DRIVE", "training" if train else "test")
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.flag = "training" if train else "test"
        self.img_list = glob.glob(f"{data_root}/images/*.tif")

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.img_list)


DriveDataset("./", train=True)