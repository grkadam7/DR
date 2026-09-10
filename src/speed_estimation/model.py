"""
SpeedNet: Lightweight 1D CNN + GRU hybrid for vehicle forward speed estimation.
Takes a sliding window of 6-DOF IMU data (channels-first) and regresses
instantaneous forward speed in m/s.

Architecture:
  Conv1d(6 → 32) → BN → ReLU → Conv1d(32 → 64) → BN → ReLU
  → GRU(64 → 64, 2 layers) → Linear(64 → 1)

Designed for ONNX export with fixed-sequence length.

Addresses Requirement: SPEED-01, SPEED-02
"""

import torch
import torch.nn as nn
from typing import Tuple


class SpeedNet(nn.Module):
    """
    Lightweight CNN-GRU speed estimator.

    Input:  (batch, 6, window_size)   -- 6 IMU channels, channels-first
    Output: (batch, 1)                -- forward speed in m/s
    """

    def __init__(
        self,
        n_channels: int = 6,
        window_size: int = 50,
        conv_channels: Tuple[int, int] = (32, 64),
        gru_hidden: int = 64,
        gru_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.window_size = window_size

        # --- Convolutional front-end ---
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_channels, conv_channels[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(inplace=True),
        )

        # --- Recurrent temporal integrator ---
        self.gru = nn.GRU(
            input_size=conv_channels[1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # --- Regression head ---
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.ReLU(inplace=True),    # speed must be non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) -- batch × channels × time
        Returns:
            speed: (B, 1) in m/s
        """
        # CNN feature extraction: (B, C, T) → (B, 64, T)
        feat = self.conv_block(x)

        # Transpose for GRU: (B, 64, T) → (B, T, 64)
        feat = feat.permute(0, 2, 1)

        # GRU: take last hidden state
        out, _ = self.gru(feat)        # out: (B, T, gru_hidden)
        last = out[:, -1, :]           # (B, gru_hidden)

        return self.head(last)          # (B, 1)

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
