from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size - 1) * dilation // 2


def _split_channels(total: int, branches: int) -> list[int]:
    base = total // branches
    rem = total % branches
    return [base + (1 if i < rem else 0) for i in range(branches)]


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=_same_padding(kernel_size, dilation),
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ZNorm(nn.Module):
    """Per-sample z-normalization over the temporal axis."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std


class MultiScaleResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: Sequence[int] = (3, 5, 7, 11),
        dilations: Sequence[int] = (1, 1, 2, 2),
        stride: int = 1,
    ) -> None:
        super().__init__()
        if len(kernels) != len(dilations):
            raise ValueError("kernels and dilations must have the same length.")

        branch_channels = _split_channels(out_channels, len(kernels))
        self.branches = nn.ModuleList(
            [
                ConvBNReLU(
                    in_channels=in_channels,
                    out_channels=branch_channels[i],
                    kernel_size=kernels[i],
                    stride=stride,
                    dilation=dilations[i],
                )
                for i in range(len(kernels))
            ]
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([branch(x) for branch in self.branches], dim=1)
        y = self.fuse(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class CNN1DMultiescalaResidualBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.3,
        kernels: Sequence[int] = (3, 5, 7, 11),
        dilations: Sequence[int] = (1, 1, 2, 2),
    ) -> None:
        super().__init__()
        self.norm = ZNorm()
        self.stem = ConvBNReLU(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            dilation=1,
        )
        self.block1 = MultiScaleResidualBlock(64, 128, kernels, dilations, stride=1)
        self.block2 = MultiScaleResidualBlock(128, 192, kernels, dilations, stride=2)
        self.block3 = MultiScaleResidualBlock(192, 256, kernels, dilations, stride=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.dropout(x)


class CNN1DMultiescalaResidualMultiHead(nn.Module):
    """Shared backbone with one classification head per dataset."""

    def __init__(
        self,
        in_channels: int,
        num_classes_by_dataset: dict[str, int],
        dropout: float = 0.3,
        kernels: Sequence[int] = (3, 5, 7, 11),
        dilations: Sequence[int] = (1, 1, 2, 2),
    ) -> None:
        super().__init__()
        if not num_classes_by_dataset:
            raise ValueError("num_classes_by_dataset cannot be empty.")
        self.backbone = CNN1DMultiescalaResidualBackbone(
            in_channels=in_channels,
            dropout=dropout,
            kernels=kernels,
            dilations=dilations,
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(256, n_classes) for name, n_classes in num_classes_by_dataset.items()}
        )
        for module in self.heads.values():
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, dataset_name: str) -> torch.Tensor:
        features = self.backbone(x)
        if dataset_name not in self.heads:
            raise KeyError(f"Unknown dataset head: {dataset_name}")
        return self.heads[dataset_name](features)
