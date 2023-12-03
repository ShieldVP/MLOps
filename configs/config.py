from dataclasses import dataclass


@dataclass
class TrainConfig:
    experiment_name: str
    weights_path: str
    device: str
    epochs: int
    tracking_uri: str


@dataclass
class InferConfig:
    weights_path: str
    device: str
    outputs_path: str


@dataclass
class DataConfig:
    data_dir: str
    image_size: int
    channels: int
    num_classes: int
    channels_mean: float
    channels_std: float
    batch_size: int


@dataclass
class ModelConfig:
    max_channels: int
    conv_kernel: int
    conv_stride: int
    conv_padding: int
    pool_kernel: int
    pool_stride: int


@dataclass
class AllConfigs:
    train_config: TrainConfig
    infer_config: InferConfig
    data_config: DataConfig
    model_config: ModelConfig
