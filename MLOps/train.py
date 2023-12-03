import mlflow
import torch
import torch.nn as nn
from configs.configs import DataConfig, ModelConfig, TrainConfig
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_issues import MNISTDataIssues
from metrics import METRICS
from model import ConvNet


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: _Loss,
        config: TrainConfig,
    ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = criterion
        self.epochs = config.epochs
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        mlflow.set_tracking_uri(uri=config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    def train(self, save_model_path: str):
        self.model.to(self.device)
        self.model.train()
        with mlflow.start_run():
            mlflow.log_params({})
            for epoch in range(self.epochs):
                print(f"Epoch #{epoch + 1}")
                self._train_epoch()
        torch.save(self.model.state_dict(), save_model_path)

    def _train_epoch(self):
        for images, labels in tqdm(self.dataloader):
            outputs = self.model(images.to(self.device))
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mlflow.log_metric("loss", loss.item())
            pred_labels = torch.argmax(outputs.detach().cpu(), dim=1)
            for metric in METRICS:
                mlflow.log_metric(metric.__name__, metric(labels, pred_labels))


def main():
    train_cfg = TrainConfig()
    data_cfg = DataConfig()
    model_cfg = ModelConfig()

    trainer = Trainer(
        model=ConvNet(model_cfg, data_cfg),
        dataloader=MNISTDataIssues(data_cfg).create_train_dataloader(),
        criterion=nn.CrossEntropyLoss(),
        config=train_cfg,
    )
    trainer.train(save_model_path=train_cfg.weights_path)


if __name__ == "__main__":
    main()
