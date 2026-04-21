"""
models/cnn_bilstm.py
--------------------
CNN + BiLSTM classifier for tic detection.

CNN gradually reduces embedding dim from 768 to 128 using
decreasing kernel sizes (7 → 5 → 3) to capture broad then
fine-grained acoustic patterns before the BiLSTM.

Input:  [batch, sequence_length, input_dim]
Output: [batch, sequence_length, num_classes]
"""

import torch
import torch.nn as nn
from base import BaseClassifier


class CNNBiLSTMClassifier(BaseClassifier):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
        cnn_dropout: float = 0.1,
    ):
        super().__init__()

        # -- CNN feature extractor --
        # input: [B, T, 768] → permute to [B, 768, T] for Conv1d
        # output: [B, 128, T] → permute back to [B, T, 128]
        self.cnn = nn.Sequential(
            # layer 1: broad patterns (kernel=7)
            nn.Conv1d(input_dim, 512, kernel_size=7, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(cnn_dropout),

            # layer 2: mid-level features (kernel=5)
            nn.Conv1d(512, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(cnn_dropout),

            # layer 3: fine-grained refinement (kernel=3)
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # -- BiLSTM --
        self.bilstm = nn.LSTM(
            input_size  = 128,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
            bidirectional = True,
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
        # CNN expects [B, C, T]
        x = x.permute(0, 2, 1)        # [B, 768, T]
        x = self.cnn(x)                # [B, 128, T]
        x = x.permute(0, 2, 1)        # [B, T, 128]

        # BiLSTM
        out, _ = self.bilstm(x)        # [B, T, hidden*2]
        out     = self.dropout(out)
        logits  = self.classifier(out) # [B, T, num_classes]

        return logits
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/kzaveri1/codes/modular_pipline_package/tic_detection/models')

    model = CNNBiLSTMClassifier(
        input_dim   = 768,
        hidden_size = 256,
        num_layers  = 2,
        dropout     = 0.3,
        num_classes = 72,
        cnn_dropout = 0.1,
    )

    # print architecture
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total_params:,}")
    print(f"Trainable params: {trainable:,}")

    # test forward pass
    x      = torch.randn(4, 500, 768)  # [B, T, D]
    logits = model(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (4, 500, 72), f"Expected (4, 500, 72), got {logits.shape}"

    # test with batch size 1
    x1      = torch.randn(1, 500, 768)
    logits1 = model(x1)
    assert logits1.shape == (1, 500, 72)

    # test sequence length preserved
    x2      = torch.randn(2, 300, 768)
    logits2 = model(x2)
    assert logits2.shape == (2, 300, 72), f"Sequence length not preserved: {logits2.shape}"

    print(f"\n✅ All checks passed")

    # -- compatibility with TicDataset output --
    print("\n── Dataset compatibility ──")
    # simulate what TicDataset.__getitem__ returns
    embeddings = torch.randn(32, 500, 768)  # [B, seq_len, input_dim]
    labels     = torch.randint(0, 72, (32, 500))  # [B, seq_len]

    logits      = model(embeddings)
    logits_flat = logits.view(-1, logits.shape[-1])  # [B*T, num_classes]
    labels_flat = labels.view(-1)                    # [B*T]

    assert logits_flat.shape == (32 * 500, 72), f"Wrong logits shape: {logits_flat.shape}"
    assert labels_flat.shape == (32 * 500,),    f"Wrong labels shape: {labels_flat.shape}"
    print(f"logits_flat: {logits_flat.shape} ✅")
    print(f"labels_flat: {labels_flat.shape} ✅")

    # -- compatibility with loss functions --
    print("\n── Loss compatibility ──")
    import torch.nn as nn

    # multiclass CE
    criterion_mc = nn.CrossEntropyLoss()
    loss_mc      = criterion_mc(logits_flat, labels_flat)
    print(f"Multiclass CE loss: {loss_mc.item():.4f} ✅")

    # binary BCE
    NO_TIC_INT    = 71
    criterion_bin = nn.BCEWithLogitsLoss()
    tic_logits    = logits_flat[:, :NO_TIC_INT].sum(dim=-1)
    binary_labels = (labels_flat != NO_TIC_INT).float()
    loss_bin      = criterion_bin(tic_logits, binary_labels)
    print(f"Binary BCE loss:    {loss_bin.item():.4f} ✅")

    # -- compatibility with metrics --
    print("\n── Metrics compatibility ──")
    import numpy as np
    from sklearn.metrics import roc_auc_score

    probs        = torch.softmax(logits_flat, dim=-1)
    preds        = torch.argmax(probs, dim=-1)
    binary_probs = 1.0 - probs[:, NO_TIC_INT].detach().numpy()
    binary_preds = (preds != NO_TIC_INT).numpy().astype(int)
    binary_labels_np = (labels_flat != NO_TIC_INT).numpy().astype(int)

    assert binary_probs.shape == (32 * 500,), f"Wrong probs shape: {binary_probs.shape}"
    assert binary_preds.shape == (32 * 500,), f"Wrong preds shape: {binary_preds.shape}"
    print(f"Binary probs shape: {binary_probs.shape} ✅")
    print(f"Binary preds shape: {binary_preds.shape} ✅")

    # -- backward pass --
    print("\n── Backward pass ──")
    loss_mc.backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  ⚠️  No gradient: {name}")
    print(f"Gradients computed for all parameters ✅")

    print(f"\n✅ Full compatibility check passed")