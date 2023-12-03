import abc
from pathlib import Path

import torch
import torchvision.transforms as transforms
from configs.configs import DataConfig
from dvc.api import DVCFileSystem
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import v2


class DataIssues(abc.ABC):
    transform: v2.Transform
    train_dataset: Dataset
    test_dataset: Dataset

    BATCH_SIZE: int

    def create_train_dataloader(self) -> DataLoader:
        torch.manual_seed(0)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
        )

    def create_test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
        )


class MNISTDataIssues(DataIssues):
    def __init__(self, cfg: DataConfig):
        root_path = Path(__file__).parent.parent
        data_dir = Path(cfg.data_dir)

        # download data if haven't yet
        mnist_dir = data_dir / "MNIST"
        if not (root_path / mnist_dir).exists():
            dvc_filesystem = DVCFileSystem(str(root_path))
            dvc_filesystem.get(str(mnist_dir), str(data_dir), recursive=True)

        # transferring img to torch.tensor and normalizing it with calculated mean and std
        self.transform = v2.Compose(
            [
                transforms.ToTensor(),
                v2.Normalize((cfg.channels_mean,), (cfg.channels_std,)),
            ]
        )

        # init train and test dataset
        abs_path_data_dir = str(root_path / data_dir)
        self.train_dataset = MNIST(
            root=abs_path_data_dir,
            train=True,
            transform=self.transform,
            download=False,
        )
        self.test_dataset = MNIST(
            root=abs_path_data_dir,
            train=False,
            transform=self.transform,
            download=False,
        )

        # setting batch size for both train and test dataloaders
        self.BATCH_SIZE = cfg.batch_size
