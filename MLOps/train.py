import hydra
import mlflow
import torch
import torch.nn as nn
from configs.config import AllConfigs, DataConfig, ModelConfig, TrainConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
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
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = criterion
        self.epochs = config.epochs
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        mlflow.set_tracking_uri(uri=config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    def train(self, save_model_path: str, model_config: ModelConfig):
        self.model.to(self.device)
        self.model.train()
        with mlflow.start_run():
            params = OmegaConf.to_container(model_config)
            params.update(OmegaConf.to_container(self.config))
            mlflow.log_params(params)
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


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)
cs.store(name="data", node=DataConfig)
cs.store(name="model", node=ModelConfig)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3.2")
def main(cfg: AllConfigs):
    trainer = Trainer(
        model=ConvNet(cfg.model_config, cfg.data_config),
        dataloader=MNISTDataIssues(cfg.data_config).create_train_dataloader(),
        criterion=nn.CrossEntropyLoss(),
        config=cfg.train_config,
    )
    trainer.train(
        save_model_path=cfg.train_config.weights_path, model_config=cfg.model_config
    )


if __name__ == "__main__":
    main()
