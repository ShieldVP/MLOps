from typing import Tuple, Dict
from data import MNISTDataIssues
from model import ConvNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
from os import path


def apply_model(model: nn.Module, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    preds = []
    answers = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            _, pred_classes = torch.max(outputs.cpu(), dim=1)
            preds.append(pred_classes)
            answers.append(labels)

    return torch.cat(preds), torch.cat(answers)


def compute_metrics(preds: torch.Tensor, answers: torch.Tensor) -> Dict[str, float]:
    return {
        'Accuracy': accuracy_score(answers, preds),
    }


if __name__ == '__main__':
    trained_model = ConvNet()
    trained_model.load_state_dict(torch.load(path.join('weights', 'chkp.pt')))
    test_dataloader = MNISTDataIssues().create_test_dataloader()

    predictions, real_answers = apply_model(trained_model, test_dataloader)

    pd.Series(predictions).to_csv(path.join('outputs', 'predictions.csv'))
    metrics = compute_metrics(predictions, real_answers)
    print(metrics)
