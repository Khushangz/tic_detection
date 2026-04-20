"""
stages/s06_train.py
-------------------
Stage 06: Train

Trains a BiLSTM classifier for tic detection.
Single model with 72 output classes (71 tic types + no-tic).
At eval time collapses to binary for binary metrics.

Reports per epoch:
    - train/val loss
    - binary AUROC, binary F1
    - multiclass AUROC (one-vs-rest), multiclass F1 (macro)

Saves:
    outputs/runs/{exp_name}/best.pt
    outputs/runs/{exp_name}/config.json
    outputs/runs/{exp_name}/metrics.json
    outputs/runs/{exp_name}/plots/

Inputs:
    configs/paths.yaml
    configs/model.yaml
    configs/label_config.json
    outputs/splits/file_split/
    outputs/splits/file_split/filter_report.json
"""

import os
import sys
from pathlib import Path
import json
import yaml
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOME_DIR = os.environ.get(
    "HOME_DIR",
    "/home/kzaveri1/codes/modular_pipline_package/tic_detection"
)
sys.path.insert(0, HOME_DIR)
sys.path.insert(0, os.path.join(HOME_DIR, "stages"))
sys.path.insert(0, os.path.join(HOME_DIR, "models"))
sys.path.insert(0, os.path.join(HOME_DIR, "utils"))

with open(os.path.join(HOME_DIR, "configs", "paths.yaml"), "r") as f:
    paths_cfg = yaml.safe_load(f)

with open(os.path.join(HOME_DIR, "configs", "model.yaml"), "r") as f:
    model_cfg = yaml.safe_load(f)

with open(os.path.join(HOME_DIR, "configs", "label_config.json"), "r") as f:
    label_config = json.load(f)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NO_TIC_INT = label_config["type_to_int"][str(label_config["no_tic_label"])]
def _build_model() -> nn.Module:
    """
    Build BiLSTM model from model.yaml config.
    Input dim inferred from label_config.
    """
    from bilstm import BiLSTMClassifier

    cfg = model_cfg["bilstm"]

    model = BiLSTMClassifier(
        input_dim   = 768,                          # WavLM base embedding dim
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg["num_layers"],
        dropout     = cfg["dropout"],
        num_classes = label_config["num_classes"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[s06] Model: BiLSTM | params={total_params:,} | trainable={trainable:,}")

    return model.to(DEVICE)

def _build_dataloaders(exp_dir: Path) -> tuple:
    """
    Build train and val DataLoaders using TicDataset and sampler.

    Returns:
        train_loader, val_loader
    """
    from s05_dataset import TicDataset
    from sampler import BatchedOversampleSampler, BatchedUndersampleSampler

    output_dir         = Path(paths_cfg["output_dir"])
    embeddings_dir     = paths_cfg["new_embeddings_dir"]
    filter_report_path = str(output_dir / "splits" / "file_split" / "filter_report.json")
    label_config_path  = os.path.join(HOME_DIR, "configs", "label_config.json")

    # -- train dataset --
    cache_dir = paths_cfg.get("cache_dir", None)

    train_dataset = TicDataset(
        split_csv          = str(output_dir / "splits" / "file_split" / "train.csv"),
        embeddings_dir     = embeddings_dir,
        filter_report_path = filter_report_path,
        label_config_path  = label_config_path,
        sequence_length    = model_cfg["sequence_length"],
        sequence_stride    = model_cfg["train_stride"],
        cache_dir          = cache_dir,
    )

    val_dataset = TicDataset(
        split_csv          = str(output_dir / "splits" / "file_split" / "val.csv"),
        embeddings_dir     = embeddings_dir,
        filter_report_path = filter_report_path,
        label_config_path  = label_config_path,
        sequence_length    = model_cfg["sequence_length"],
        sequence_stride    = model_cfg["eval_stride"],
        cache_dir          = cache_dir,
    )


    # -- sampler --
    strategy = model_cfg["imbalance_strategy"]
    if strategy == "batched_oversample":
        sampler = BatchedOversampleSampler(
            train_dataset,
            batch_size = model_cfg["batch_size"],
            no_tic_int = NO_TIC_INT,
            seed       = 42,
        )
    elif strategy == "batched_undersample":
        sampler = BatchedUndersampleSampler(
            train_dataset,
            batch_size        = model_cfg["batch_size"],
            no_tic_int        = NO_TIC_INT,
            undersample_ratio = model_cfg["undersample_ratio"],
            seed              = 42,
        )
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size  = model_cfg["batch_size"],
        sampler     = sampler,
        shuffle     = sampler is None,
        num_workers = 8,
        pin_memory  = True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = model_cfg["batch_size"],
        shuffle     = False,
        num_workers = 8,
        pin_memory  = True,
    )

    print(f"[s06] Train batches: {len(train_loader)}")
    print(f"[s06] Val batches:   {len(val_loader)}")

    return train_loader, val_loader

def _compute_metrics(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    all_probs: np.ndarray,
) -> dict:
    """
    Compute binary and multiclass metrics.

    Binary: collapse all non-no-tic predictions to positive (1)
    Multiclass: per-class one-vs-rest AUROC, macro F1

    Args:
        all_labels: [N] true frame labels (int)
        all_preds:  [N] predicted class indices (int)
        all_probs:  [N, num_classes] softmax probabilities

    Returns:
        dict with binary and multiclass metrics
    """
    # -- binary collapse --
    binary_labels = (all_labels != NO_TIC_INT).astype(int)
    binary_preds  = (all_preds  != NO_TIC_INT).astype(int)
    binary_probs  = 1.0 - all_probs[:, NO_TIC_INT]  # prob of being a tic

    # -- binary metrics --
    binary_f1   = f1_score(binary_labels, binary_preds, zero_division=0)
    try:
        binary_auroc = roc_auc_score(binary_labels, binary_probs)
    except ValueError:
        binary_auroc = 0.0

    # -- multiclass metrics --
    # only evaluate on frames that belong to included classes
    tic_mask = all_labels != NO_TIC_INT
    if tic_mask.sum() > 0:
        mc_labels = all_labels[tic_mask]
        mc_preds  = all_preds[tic_mask]
        mc_probs  = all_probs[tic_mask]

        mc_f1 = f1_score(mc_labels, mc_preds, average="macro", zero_division=0)

        try:
            # one-vs-rest AUROC for classes present in labels
            present_classes = np.unique(mc_labels)
            if len(present_classes) > 1:
                mc_auroc = roc_auc_score(
                    mc_labels,
                    mc_probs[:, present_classes],
                    multi_class="ovr",
                    labels=present_classes,
                )
            else:
                mc_auroc = 0.0
        except ValueError:
            mc_auroc = 0.0
    else:
        mc_f1    = 0.0
        mc_auroc = 0.0

    return {
        "binary_f1":    round(binary_f1,    4),
        "binary_auroc": round(binary_auroc, 4),
        "mc_f1":        round(mc_f1,        4),
        "mc_auroc":     round(mc_auroc,     4),
    }
def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """
    Run one training epoch.

    Returns:
        avg_loss: float
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for embeddings, labels in loader:
        embeddings = embeddings.to(DEVICE)  # [B, L, D]
        labels     = labels.to(DEVICE)      # [B, L]

        optimizer.zero_grad()

        logits = model(embeddings)          # [B, L, num_classes]

        # reshape for loss: [B*L, num_classes] and [B*L]
        logits_flat = logits.view(-1, logits.shape[-1])
        labels_flat = labels.view(-1)

        loss = criterion(logits_flat, labels_flat)
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)

def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, dict]:
    """
    Run one validation epoch.

    Returns:
        avg_loss: float
        metrics:  dict with binary and multiclass metrics
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    all_labels = []
    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(DEVICE)  # [B, L, D]
            labels     = labels.to(DEVICE)      # [B, L]

            logits = model(embeddings)           # [B, L, num_classes]

            # reshape for loss
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)

            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()
            n_batches  += 1

            # probabilities and predictions
            probs = torch.softmax(logits_flat, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_labels.append(labels_flat.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    all_probs  = np.concatenate(all_probs)

    metrics = _compute_metrics(all_labels, all_preds, all_probs)
    avg_loss = total_loss / max(n_batches, 1)

    return avg_loss, metrics
def _save_plots(history: dict, plots_dir: Path) -> None:
    """
    Generate and save training plots.

    Plots:
        - train/val loss curve
        - binary AUROC curve
        - multiclass AUROC curve
        - binary F1 curve
        - multiclass F1 curve

    Args:
        history:   dict of lists, one value per epoch
        plots_dir: directory to save plots
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # -- style --
    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
    })

    # -- loss curve --
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], label="train", color="#2563a8", linewidth=1.5)
    ax.plot(epochs, history["val_loss"],   label="val",   color="#c05621", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train / Val Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "loss.png", dpi=150)
    plt.close(fig)

    # -- binary AUROC --
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["val_binary_auroc"], color="#2563a8", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.set_title("Validation Binary AUROC")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plots_dir / "binary_auroc.png", dpi=150)
    plt.close(fig)

    # -- multiclass AUROC --
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["val_mc_auroc"], color="#276749", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.set_title("Validation Multiclass AUROC (OvR)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plots_dir / "mc_auroc.png", dpi=150)
    plt.close(fig)

    # -- binary F1 --
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["val_binary_f1"], color="#2563a8", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title("Validation Binary F1")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plots_dir / "binary_f1.png", dpi=150)
    plt.close(fig)

    # -- multiclass F1 --
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["val_mc_f1"], color="#276749", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 (macro)")
    ax.set_title("Validation Multiclass F1 (macro)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plots_dir / "mc_f1.png", dpi=150)
    plt.close(fig)

    print(f"[s06] Plots saved to: {plots_dir}")
def run_train(exp_name: str = "exp_01") -> None:
    """
    Run the full training loop.

    Args:
        exp_name: experiment name, used for output directory
    """
    # -- setup logging --
    output_dir = Path(paths_cfg["output_dir"])
    exp_dir    = output_dir / "runs" / exp_name
    plots_dir  = exp_dir / "plots"
    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = exp_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger(__name__)
    log.info(f"[s06] Experiment: {exp_name}")
    log.info(f"[s06] Device: {DEVICE}")

    # -- save config --
    with open(exp_dir / "config.json", "w") as f:
        json.dump(model_cfg, f, indent=2)

    # -- build dataloaders --
    train_loader, val_loader = _build_dataloaders(exp_dir)

    # -- build model --
    model = _build_model()

    # -- loss function --
    if model_cfg.get("use_class_weights", False):
        # compute inverse frequency weights from label_config
        class_counts = label_config.get("class_counts", {})
        total        = sum(class_counts.values())
        weights      = torch.ones(label_config["num_classes"])
        for type_str, count in class_counts.items():
            idx = label_config["type_to_int"].get(type_str)
            if idx is not None and count > 0:
                weights[idx] = total / (label_config["num_classes"] * count)
        weights = weights.to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        log.info(f"[s06] Using class weights")
    else:
        criterion = nn.CrossEntropyLoss()
        log.info(f"[s06] No class weights")

    # -- optimizer --
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_cfg["lr"]))

    # -- scheduler --
    if model_cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=model_cfg["epochs"]
        )
    elif model_cfg["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    elif model_cfg["scheduler"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
    else:
        scheduler = None

    # -- training loop --
    best_val_auroc = 0.0
    best_epoch     = 0
    history = {
        "train_loss":       [],
        "val_loss":         [],
        "val_binary_auroc": [],
        "val_binary_f1":    [],
        "val_mc_auroc":     [],
        "val_mc_f1":        [],
    }

    for epoch in range(1, model_cfg["epochs"] + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion)
        val_loss, metrics = _val_epoch(model, val_loader, criterion)

        # -- scheduler step --
        if scheduler is not None:
            if model_cfg["scheduler"] == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # -- record history --
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_binary_auroc"].append(metrics["binary_auroc"])
        history["val_binary_f1"].append(metrics["binary_f1"])
        history["val_mc_auroc"].append(metrics["mc_auroc"])
        history["val_mc_f1"].append(metrics["mc_f1"])

        log.info(
            f"[s06] Epoch {epoch:3d}/{model_cfg['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"bin_auroc={metrics['binary_auroc']:.4f} | "
            f"bin_f1={metrics['binary_f1']:.4f} | "
            f"mc_auroc={metrics['mc_auroc']:.4f} | "
            f"mc_f1={metrics['mc_f1']:.4f}"
        )

        # -- save best model --
        if metrics["binary_auroc"] > best_val_auroc:
            best_val_auroc = metrics["binary_auroc"]
            best_epoch     = epoch
            torch.save(model.state_dict(), exp_dir / "best.pt")
            log.info(f"[s06] ✅ New best model saved (epoch {epoch}, binary_auroc={best_val_auroc:.4f})")

    # -- save metrics --
    metrics_out = {
        "best_epoch":      best_epoch,
        "best_val_auroc":  best_val_auroc,
        "history":         history,
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # -- save plots --
    _save_plots(history, plots_dir)

    log.info(f"[s06] Training complete.")
    log.info(f"[s06] Best epoch: {best_epoch} | Binary AUROC: {best_val_auroc:.4f}")
    log.info(f"[s06] Results saved to: {exp_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="exp_01")
    args = parser.parse_args()
    run_train(exp_name=args.exp)