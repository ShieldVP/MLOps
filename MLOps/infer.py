from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from configs.configs import DataConfig, InferConfig, ModelConfig
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


def main():
    infer_cfg = InferConfig()
    data_cfg = DataConfig()
    model_cfg = ModelConfig()

    trained_model = ConvNet(model_cfg, data_cfg)
    trained_model.load_state_dict(torch.load(infer_cfg.weights_path))
    test_dataloader = MNISTDataIssues(data_cfg).create_test_dataloader()

    predictions, real_answers = apply_model(trained_model, test_dataloader, infer_cfg)

    pd.Series(predictions).to_csv(infer_cfg.outputs_path)
    metrics = compute_metrics(predictions, real_answers)
    print(metrics)


if __name__ == "__main__":
    main()
