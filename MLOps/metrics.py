import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def micro_precision_score(y_true, y_pred):
    return precision_score(y_true, y_pred, average="micro", zero_division=np.nan)


def macro_precision_score(y_true, y_pred):
    return precision_score(y_true, y_pred, average="macro", zero_division=np.nan)


def micro_recall_score(y_true, y_pred):
    return recall_score(y_true, y_pred, average="micro", zero_division=np.nan)


def macro_recall_score(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro", zero_division=np.nan)


METRICS = [
    accuracy_score,
    macro_recall_score,
    micro_recall_score,
    macro_precision_score,
    micro_precision_score,
]
