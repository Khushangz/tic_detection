"""
stages/s07_eval.py
------------------
Stage 07: Evaluation

Loads best checkpoint and runs inference on test set.
Applies optional temporal voting over frame predictions.
Reports binary and multiclass metrics, per-group breakdown,
and confusion matrix.

Inputs:
    configs/paths.yaml
    configs/eval.yaml
    configs/label_config.json
    outputs/runs/{exp_name}/best.pt
    outputs/runs/{exp_name}/config.json
    outputs/splits/{test_split}/test.csv
    outputs/splits/{test_split}/filter_report.json

Outputs:
    outputs/runs/{exp_name}/eval/results.json
    outputs/runs/{exp_name}/eval/confusion_matrix.png
    outputs/runs/{exp_name}/eval/per_group_metrics.csv
    outputs/runs/{exp_name}/eval/predictions.csv  (if save_predictions=true)
"""

import os
import sys
from pathlib import Path
import json
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix as sk_confusion_matrix,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

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

with open(os.path.join(HOME_DIR, "configs", "eval.yaml"), "r") as f:
    eval_cfg = yaml.safe_load(f)

with open(os.path.join(HOME_DIR, "configs", "label_config.json"), "r") as f:
    label_config = json.load(f)

with open(os.path.join(HOME_DIR, "configs", "model.yaml"), "r") as f:
    model_cfg = yaml.safe_load(f)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NO_TIC_INT = label_config["type_to_int"][str(label_config["no_tic_label"])]


def _load_model(checkpoint_path: Path, exp_dir: Path) -> nn.Module:
    """
    Load model from checkpoint using the config saved with the experiment.
    Falls back to current model.yaml if no experiment config found.
    """
    from factory import get_model

    # always use the config the model was trained with
    exp_config_path = exp_dir / "config.json"
    if exp_config_path.exists():
        with open(exp_config_path) as f:
            exp_model_cfg = json.load(f)
        print(f"[s07] Using experiment config: {exp_config_path}")
    else:
        exp_model_cfg = model_cfg
        print(f"[s07] ⚠️  No experiment config found, falling back to model.yaml")

    model = get_model(
        model_cfg   = exp_model_cfg,
        num_classes = label_config["num_classes"],
        input_dim   = 768,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    )
    model.eval()
    model.to(DEVICE)
    print(f"[s07] Model loaded from: {checkpoint_path}")
    return model, exp_model_cfg


def _apply_voting(
    preds: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not eval_cfg["voting"]["enabled"]:
        return preds, probs

    window   = eval_cfg["voting"]["window"]
    strategy = eval_cfg["voting"]["strategy"]
    n        = len(preds)
    half     = window // 2

    smoothed_preds = preds.copy()
    smoothed_probs = probs.copy()

    for i in range(n):
        start = max(0, i - half)
        end   = min(n, i + half + 1)

        if strategy == "majority":
            window_preds      = preds[start:end]
            counts            = np.bincount(window_preds, minlength=label_config["num_classes"])
            smoothed_preds[i] = np.argmax(counts)
            smoothed_probs[i] = counts / counts.sum()

        elif strategy == "mean_prob":
            window_probs      = probs[start:end]
            mean_prob         = window_probs.mean(axis=0)
            smoothed_probs[i] = mean_prob
            smoothed_preds[i] = np.argmax(mean_prob)

    return smoothed_preds, smoothed_probs


def _compute_metrics(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    all_probs: np.ndarray,
) -> dict:
    # -- binary --
    binary_labels = (all_labels != NO_TIC_INT).astype(int)
    binary_preds  = (all_preds  != NO_TIC_INT).astype(int)
    binary_probs  = 1.0 - all_probs[:, NO_TIC_INT]

    binary_f1        = f1_score(binary_labels, binary_preds, zero_division=0)
    binary_precision = precision_score(binary_labels, binary_preds, zero_division=0)
    binary_recall    = recall_score(binary_labels, binary_preds, zero_division=0)
    try:
        binary_auroc = roc_auc_score(binary_labels, binary_probs)
    except ValueError:
        binary_auroc = 0.0

    # -- multiclass --
    tic_mask = all_labels != NO_TIC_INT
    if tic_mask.sum() > 0:
        mc_labels = all_labels[tic_mask]
        mc_preds  = all_preds[tic_mask]
        mc_probs  = all_probs[tic_mask]

        mc_f1        = f1_score(mc_labels, mc_preds, average="macro", zero_division=0)
        mc_precision = precision_score(mc_labels, mc_preds, average="macro", zero_division=0)
        mc_recall    = recall_score(mc_labels, mc_preds, average="macro", zero_division=0)

        try:
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
        mc_f1 = mc_precision = mc_recall = mc_auroc = 0.0

    return {
        "binary_auroc":     round(binary_auroc,     4),
        "binary_f1":        round(binary_f1,        4),
        "binary_precision": round(binary_precision, 4),
        "binary_recall":    round(binary_recall,    4),
        "mc_auroc":         round(mc_auroc,         4),
        "mc_f1":            round(mc_f1,            4),
        "mc_precision":     round(mc_precision,     4),
        "mc_recall":        round(mc_recall,        4),
    }


def _compute_per_group_metrics(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    all_probs: np.ndarray,
) -> pd.DataFrame:
    int_to_type   = {int(k): v for k, v in label_config["int_to_type"].items()}
    type_to_group = {int(k): v for k, v in label_config["type_to_group"].items()}

    int_to_group = {}
    for int_label, type_val in int_to_type.items():
        if type_val != label_config["no_tic_label"]:
            int_to_group[int_label] = type_to_group.get(int(type_val), "unknown")

    tic_mask   = all_labels != NO_TIC_INT
    tic_labels = all_labels[tic_mask]
    tic_preds  = all_preds[tic_mask]
    tic_probs  = all_probs[tic_mask]

    groups_present = sorted(set(
        int_to_group.get(l, "unknown") for l in np.unique(tic_labels)
    ))

    rows = []
    for group in groups_present:
        group_ints          = [i for i, g in int_to_group.items() if g == group]
        group_binary_labels = np.isin(tic_labels, group_ints).astype(int)
        group_binary_preds  = np.isin(tic_preds,  group_ints).astype(int)
        group_probs         = tic_probs[:, group_ints].sum(axis=1)

        n_positive = group_binary_labels.sum()
        n_total    = len(group_binary_labels)

        f1        = f1_score(group_binary_labels, group_binary_preds, zero_division=0)
        precision = precision_score(group_binary_labels, group_binary_preds, zero_division=0)
        recall    = recall_score(group_binary_labels, group_binary_preds, zero_division=0)

        try:
            auroc = roc_auc_score(group_binary_labels, group_probs) if n_positive > 0 else 0.0
        except ValueError:
            auroc = 0.0

        rows.append({
            "group":      group,
            "n_frames":   n_total,
            "n_positive": int(n_positive),
            "auroc":      round(auroc,     4),
            "f1":         round(f1,        4),
            "precision":  round(precision, 4),
            "recall":     round(recall,    4),
        })

    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)


def _save_confusion_matrix(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    output_path: Path,
) -> None:
    tic_mask   = all_labels != NO_TIC_INT
    tic_labels = all_labels[tic_mask]
    tic_preds  = all_preds[tic_mask]

    if len(tic_labels) == 0:
        print(f"[s07] ⚠️  No tic frames found, skipping confusion matrix")
        return

    int_to_type   = {int(k): v for k, v in label_config["int_to_type"].items()}
    type_to_group = {int(k): v for k, v in label_config["type_to_group"].items()}

    present_classes = sorted(np.unique(np.concatenate([tic_labels, tic_preds])))
    class_names = []
    for c in present_classes:
        type_val   = int_to_type.get(c, c)
        group_name = type_to_group.get(int(type_val), str(type_val)) if type_val != label_config["no_tic_label"] else "no-tic"
        class_names.append(f"{group_name}\n({type_val})")

    cm       = sk_confusion_matrix(tic_labels, tic_preds, labels=present_classes)
    cm_norm  = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_norm, row_sums, where=row_sums != 0)

    n       = len(present_classes)
    figsize = max(10, n * 0.5)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix (tic frames only, row-normalized)", fontsize=12)

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 0.5 else "black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[s07] Confusion matrix saved to: {output_path}")


def run_eval(exp_name: str = "exp_01", eval_suffix: str = "") -> None:
    output_dir = Path(paths_cfg["output_dir"])
    exp_dir    = output_dir / "runs" / exp_name
    eval_dir = exp_dir / f"eval{eval_suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    log_path = eval_dir / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger(__name__)
    log.info(f"[s07] Evaluating experiment: {exp_name}")
    log.info(f"[s07] Device: {DEVICE}")

    # -- load model using experiment's saved config --
    checkpoint_path = exp_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"[s07] Checkpoint not found: {checkpoint_path}")
    model, exp_model_cfg = _load_model(checkpoint_path, exp_dir)

    # -- build test dataloader --
    from s05_dataset import TicDataset

    test_split     = eval_cfg.get("test_split", "file_split")
    split_dir      = output_dir / "splits" / test_split
    embeddings_dir = paths_cfg["new_embeddings_dir"]
    cache_dir      = paths_cfg.get("cache_dir", None)

    log.info(f"[s07] Loading test dataset from: {split_dir}")
    test_dataset = TicDataset(
        split_csv          = str(split_dir / "test.csv"),
        embeddings_dir     = embeddings_dir,
        filter_report_path = str(split_dir / "filter_report.json"),
        label_config_path  = os.path.join(HOME_DIR, "configs", "label_config.json"),
        sequence_length    = exp_model_cfg["sequence_length"],
        sequence_stride    = exp_model_cfg["eval_stride"],
        cache_dir          = cache_dir,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = exp_model_cfg["batch_size"],
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
    )
    log.info(f"[s07] Test batches: {len(test_loader)}")

    # -- inference --
    log.info(f"[s07] Running inference...")
    all_labels = []
    all_preds  = []
    all_probs  = []

    model.eval()
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings  = embeddings.to(DEVICE)
            labels      = labels.to(DEVICE)

            logits      = model(embeddings)
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)

            probs = torch.softmax(logits_flat, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_labels.append(labels_flat.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    all_probs  = np.concatenate(all_probs)

    log.info(f"[s07] Total frames: {len(all_labels):,}")

    # -- temporal voting --
    if eval_cfg["voting"]["enabled"]:
        log.info(
            f"[s07] Applying {eval_cfg['voting']['strategy']} voting "
            f"(window={eval_cfg['voting']['window']})"
        )
        all_preds, all_probs = _apply_voting(all_preds, all_probs)

    # -- compute metrics --
    log.info(f"[s07] Computing metrics...")
    metrics = _compute_metrics(all_labels, all_preds, all_probs)

    log.info(f"\n[s07] ── Results ──")
    log.info(f"[s07]   binary_auroc:     {metrics['binary_auroc']}")
    log.info(f"[s07]   binary_f1:        {metrics['binary_f1']}")
    log.info(f"[s07]   binary_precision: {metrics['binary_precision']}")
    log.info(f"[s07]   binary_recall:    {metrics['binary_recall']}")
    log.info(f"[s07]   mc_auroc:         {metrics['mc_auroc']}")
    log.info(f"[s07]   mc_f1:            {metrics['mc_f1']}")
    log.info(f"[s07]   mc_precision:     {metrics['mc_precision']}")
    log.info(f"[s07]   mc_recall:        {metrics['mc_recall']}")

    # -- per group metrics --
    per_group_df   = _compute_per_group_metrics(all_labels, all_preds, all_probs)
    per_group_path = eval_dir / "per_group_metrics.csv"
    per_group_df.to_csv(per_group_path, index=False)
    log.info(f"\n[s07] ── Per Group Metrics ──")
    log.info(f"\n{per_group_df.to_string(index=False)}")

    # -- confusion matrix --
    _save_confusion_matrix(
        all_labels, all_preds,
        eval_dir / "confusion_matrix.png"
    )

    # -- save predictions --
    if eval_cfg.get("save_predictions", True):
        pred_df = pd.DataFrame({
            "true_label":  all_labels,
            "pred_label":  all_preds,
            "is_tic_true": (all_labels != NO_TIC_INT).astype(int),
            "is_tic_pred": (all_preds  != NO_TIC_INT).astype(int),
            "tic_prob":    1.0 - all_probs[:, NO_TIC_INT],
        })
        pred_df.to_csv(eval_dir / "predictions.csv", index=False)
        log.info(f"[s07] Predictions saved to: {eval_dir / 'predictions.csv'}")

    # -- save results --
    results = {
        "exp_name":      exp_name,
        "test_split":    test_split,
        "voting":        eval_cfg["voting"],
        "metrics":       metrics,
        "n_test_frames": int(len(all_labels)),
        "n_tic_frames":  int((all_labels != NO_TIC_INT).sum()),
    }
    with open(eval_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n[s07] Results saved to: {eval_dir}")
    log.info(f"[s07] Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="exp_01")
    parser.add_argument("--eval-suffix", type=str, default="", dest="eval_suffix")
    args = parser.parse_args()
    run_eval(exp_name=args.exp, eval_suffix=args.eval_suffix)