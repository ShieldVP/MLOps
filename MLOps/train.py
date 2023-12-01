from os import path

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import MNISTDataIssues
from model import ConvNet


class Trainer:
    def __init__(
        self, model: nn.Module, dataloader: DataLoader, criterion: _Loss, epochs: int = 3
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = criterion
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, save_model_path: str):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            print(f"Epoch #{epoch + 1}")
            self.train_loop()
        torch.save(self.model.state_dict(), save_model_path)

    def train_loop(self):
        for images, labels in tqdm(self.dataloader):
            outputs = self.model(images.to(self.device))
            loss = self.criterion(outputs, labels.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    trainer = Trainer(
        model=ConvNet(),
        dataloader=MNISTDataIssues().create_train_dataloader(),
        criterion=nn.CrossEntropyLoss(),
    )
    trainer.train(path.join("weights", "chkp.pt"))
