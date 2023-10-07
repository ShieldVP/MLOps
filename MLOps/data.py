import abc

import torchvision.transforms as transforms
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
    DATA_PATH = "../data/"

    train_dataset = MNIST(
        root=DATA_PATH,
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = MNIST(
        root=DATA_PATH,
        train=False,
        transform=transform,
    )

    BATCH_SIZE = 8
