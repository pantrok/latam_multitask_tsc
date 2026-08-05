from __future__ import annotations

import torch
from torch import nn


class ZNorm(nn.Module):
    """Per-sample z-normalization over the temporal axis."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std


class LSTMMultitarea(nn.Module):
    """Shared LSTM encoder with one classification head per dataset."""

    def __init__(
        self,
        in_channels: int,
        num_classes_by_dataset: dict[str, int],
        hidden_size: int = 128,
        num_layers: int = 2,
        lstm_dropout: float = 0.2,
        head_dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if not num_classes_by_dataset:
            raise ValueError("num_classes_by_dataset cannot be empty.")

        self.norm = ZNorm()
        self.encoder = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        feat_dim = hidden_size * (2 if bidirectional else 1)
        self.head_dropout = nn.Dropout(head_dropout)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(feat_dim, n_classes) for name, n_classes in num_classes_by_dataset.items()}
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.encoder.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for layer in self.heads.values():
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Input expected as (B, C, T); LSTM consumes (B, T, C).
        x = self.norm(x)
        x = x.transpose(1, 2)
        out, _ = self.encoder(x)
        features = out.mean(dim=1)
        return self.head_dropout(features)

    def forward(self, x: torch.Tensor, dataset_name: str) -> torch.Tensor:
        if dataset_name not in self.heads:
            raise KeyError(f"Unknown dataset head: {dataset_name}")
        features = self.encode(x)
        return self.heads[dataset_name](features)
