"""
models/base.py
--------------
Abstract base class for all classifiers in the pipeline.
All models inherit from this and implement forward().
"""

import torch
import torch.nn as nn
from pathlib import Path


class BaseClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            logits: [batch, seq_len, num_classes]
        """
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            class indices: [batch, seq_len]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            probabilities: [batch, seq_len, num_classes]
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

    def save(self, path: str | Path) -> None:
        """Save model state dict."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[model] Saved to: {path}")

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "BaseClassifier":
        """Load model from state dict."""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model