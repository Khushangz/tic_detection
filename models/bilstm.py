"""
models/bilstm.py
----------------
BiLSTM classifier for tic detection.

Input:  [batch, sequence_length, input_dim]
Output: [batch, sequence_length, num_classes]

Frame-level predictions — one class per 20ms frame.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            logits: [batch, seq_len, num_classes]
        """
        out, _ = self.bilstm(x)        # [batch, seq_len, hidden*2]
        out     = self.dropout(out)
        logits  = self.classifier(out) # [batch, seq_len, num_classes]
        return logits