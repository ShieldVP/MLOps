import torch
import torch.nn as nn
from configs.config import DataConfig, ModelConfig


class ConvNet(nn.Module):
    def __init__(self, model_cfg: ModelConfig, data_cfg: DataConfig):
        super(ConvNet, self).__init__()

        self.model_cfg = model_cfg

        torch.manual_seed(0)
        self.layer1 = self._create_layer(data_cfg.channels, model_cfg.max_channels // 2)
        self.layer2 = self._create_layer(
            model_cfg.max_channels // 2, model_cfg.max_channels
        )
        self.drop_out = nn.Dropout()
        self.fc = nn.Linear(
            model_cfg.max_channels
            * self._calculate_tensor_side(
                self._calculate_tensor_side(data_cfg.image_size)
            )
            ** 2,
            data_cfg.num_classes,
        )

    def _create_layer(self, channels_in: int, channels_out: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=self.model_cfg.conv_kernel,
                stride=self.model_cfg.conv_stride,
                padding=self.model_cfg.conv_padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=self.model_cfg.pool_kernel, stride=self.model_cfg.pool_stride
            ),
        )

    def _calculate_tensor_side(self, input_side: int) -> int:
        """Calculate tensor side after one basic layer"""
        after_conv_side = (
            input_side + 2 * self.model_cfg.conv_padding - self.model_cfg.conv_kernel
        ) // self.model_cfg.conv_stride + 1
        after_pool_side = (
            after_conv_side - self.model_cfg.pool_stride
        ) // self.model_cfg.pool_stride + 1
        return after_pool_side

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc(out)
        return out
