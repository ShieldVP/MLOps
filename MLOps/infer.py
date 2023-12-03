from typing import Dict, Tuple

import hydra
import pandas as pd
import torch
import torch.nn as nn
from configs.config import AllConfigs, DataConfig, InferConfig, ModelConfig
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_issues import MNISTDataIssues
from metrics import METRICS
from model import ConvNet


def apply_model(
    model: nn.Module, dataloader: DataLoader, cfg: InferConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = []
    answers = []
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for images, labels in tqdm(dataloader):
            outputs = model(images.to(device))
            pred_classes = torch.argmax(outputs.cpu(), dim=1)
            preds.append(pred_classes)
            answers.append(labels)

    return torch.cat(preds), torch.cat(answers)


def compute_metrics(preds: torch.Tensor, answers: torch.Tensor) -> Dict[str, float]:
    return {metric.__name__: metric(answers, preds) for metric in METRICS}


cs = ConfigStore.instance()
cs.store(name="infer", node=InferConfig)
cs.store(name="data", node=DataConfig)
cs.store(name="model", node=ModelConfig)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3.2")
def main(cfg: AllConfigs):
    trained_model = ConvNet(cfg.model_config, cfg.data_config)
    trained_model.load_state_dict(torch.load(cfg.infer_config.weights_path))
    test_dataloader = MNISTDataIssues(cfg.data_config).create_test_dataloader()

    predictions, real_answers = apply_model(
        trained_model, test_dataloader, cfg.infer_config
    )

    pd.Series(predictions).to_csv(cfg.infer_config.outputs_path)
    metrics = compute_metrics(predictions, real_answers)
    print(metrics)


if __name__ == "__main__":
    main()
