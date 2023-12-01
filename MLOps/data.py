import abc
from pathlib import Path

import torchvision.transforms as transforms
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
    transform = v2.Compose(
        [
            transforms.ToTensor(),
            v2.Normalize((0.1307,), (0.3081,)),
        ]
    )

    root_dir = Path("/Users/yabatrakov/Desktop/MLOps")
    data_dir = Path("data")
    mnist_dir = data_dir / "MNIST"
    if not mnist_dir.exists():
        dvc_filesystem = DVCFileSystem(str(root_dir))
        dvc_filesystem.get(str(mnist_dir), str(data_dir), recursive=True)

    abs_data_dir = str(root_dir / "data")
    train_dataset = MNIST(
        root=abs_data_dir,
        train=True,
        transform=transform,
        download=False,
    )
    test_dataset = MNIST(
        root=abs_data_dir,
        train=False,
        transform=transform,
        download=False,
    )

    BATCH_SIZE = 8
