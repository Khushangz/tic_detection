"""
models/linear_probe.py
----------------------
Linear probe classifier for tic detection.

Single linear layer directly on top of WavLM embeddings.
Minimal parameters (~1.5K) to serve as a strong baseline
and diagnose whether overfitting is a model complexity issue.

Input:  [batch, sequence_length, input_dim]
Output: [batch, sequence_length, num_classes]
"""

import torch
import torch.nn as nn
from base import BaseClassifier


class LinearProbe(BaseClassifier):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        **kwargs,  # absorb unused args from factory
    ):
        super().__init__()

        self.dropout    = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(input_dim, num_classes)

        total = sum(p.numel() for p in self.parameters())
        print(f"[linear_probe] params={total:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            logits: [batch, seq_len, num_classes]
        """
        x = self.dropout(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import numpy as np
    sys.path.insert(0, '/home/kzaveri1/codes/modular_pipline_package/tic_detection/models')

    model = LinearProbe(input_dim=768, num_classes=72, dropout=0.1)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params:     {total_params:,}")

    # forward pass
    x      = torch.randn(4, 500, 768)
    logits = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (4, 500, 72)

    # dataset compatibility
    print("\n── Dataset compatibility ──")
    embeddings  = torch.randn(32, 500, 768)
    labels      = torch.randint(0, 72, (32, 500))
    logits      = model(embeddings)
    logits_flat = logits.view(-1, logits.shape[-1])
    labels_flat = labels.view(-1)
    assert logits_flat.shape == (32 * 500, 72)
    assert labels_flat.shape == (32 * 500,)
    print(f"logits_flat: {logits_flat.shape} ✅")
    print(f"labels_flat: {labels_flat.shape} ✅")

    # loss compatibility
    print("\n── Loss compatibility ──")
    criterion_mc  = nn.CrossEntropyLoss()
    loss_mc       = criterion_mc(logits_flat, labels_flat)
    print(f"Multiclass CE loss: {loss_mc.item():.4f} ✅")

    NO_TIC_INT    = 71
    criterion_bin = nn.BCEWithLogitsLoss()
    tic_logits    = logits_flat[:, :NO_TIC_INT].sum(dim=-1)
    binary_labels = (labels_flat != NO_TIC_INT).float()
    loss_bin      = criterion_bin(tic_logits, binary_labels)
    print(f"Binary BCE loss:    {loss_bin.item():.4f} ✅")

    # backward pass
    print("\n── Backward pass ──")
    loss_mc.backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  ⚠️  No gradient: {name}")
    print(f"Gradients computed ✅")

    print(f"\n✅ All checks passed")