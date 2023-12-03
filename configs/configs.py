from dataclasses import dataclass


@dataclass
class TrainConfig:
    experiment_name: str = "Experiment name"
    weights_path: str = "weights/checkpoint.pt"
    device: str = "cuda:0"
    epochs: int = 2
    tracking_uri: str = "http://127.0.0.1:8080"


@dataclass
class InferConfig:
    weights_path: str = "weights/chkp.pt"
    device: str = "cuda:0"
    outputs_path: str = "outputs/predictions.csv"


@dataclass
class DataConfig:
    data_dir: str = "data"
    image_size: int = 28
    channels: int = 1
    num_classes: int = 10
    channels_mean: float = 0.1307
    channels_std: float = 0.3081
    batch_size: int = 8


@dataclass
class ModelConfig:
    max_channels: int = 64
    conv_kernel: int = 5
    conv_stride: int = 1
    conv_padding: int = 2
    pool_kernel: int = 2
    pool_stride: int = 2
