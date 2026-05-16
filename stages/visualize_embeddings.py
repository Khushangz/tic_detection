"""
stages/visualize_embeddings.py
-------------------------------
Visualize WavLM embeddings in 2D using t-SNE and UMAP.

Samples frames from train split, runs dimensionality reduction,
and plots tic vs no-tic separability. Also plots per-group.

Outputs:
    outputs/viz/tsne_binary.png
    outputs/viz/tsne_groups.png
    outputs/viz/umap_binary.png      (if umap installed)
    outputs/viz/umap_groups.png      (if umap installed)
    outputs/viz/pca_binary.png       (fast sanity check)
"""

import os
import sys
import json
import yaml
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

HOME_DIR = os.environ.get(
    "HOME_DIR",
    "/home/kzaveri1/codes/modular_pipline_package/tic_detection"
)
sys.path.insert(0, HOME_DIR)
sys.path.insert(0, os.path.join(HOME_DIR, "stages"))

with open(os.path.join(HOME_DIR, "configs", "paths.yaml")) as f:
    paths_cfg = yaml.safe_load(f)
with open(os.path.join(HOME_DIR, "configs", "label_config.json")) as f:
    label_config = json.load(f)

NO_TIC_INT    = label_config["type_to_int"][str(label_config["no_tic_label"])]
int_to_type   = {int(k): v for k, v in label_config["int_to_type"].items()}
type_to_group = {int(k): v for k, v in label_config["type_to_group"].items()}

# -- style --
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

COLORS = {
    "no-tic": "#9e9b93",
    "Atypical Breathing":      "#2563a8",
    "Atypical Breathing Tics": "#1d4ed8",
    "Barking":                 "#c05621",
    "Coprolalia":              "#dc2626",
    "Coughing":                "#276749",
    "DisinhibitedSpeech":      "#065f46",
    "Grunting":                "#7c3aed",
    "Hand Movements":          "#db2777",
    "Mouth Movements":         "#b45309",
    "Mouth Noises":            "#0891b2",
    "Mouth Noises - Raspberry":"#0e7490",
    "Nose Movements":          "#16a34a",
    "Other Animal Noises":     "#ea580c",
    "Sniffing":                "#9333ea",
    "Snorting":                "#e11d48",
    "Syllables":               "#0284c7",
    "Throat Clearing":         "#15803d",
}


def _sample_frames(
    split_csv: str,
    embeddings_dir: str,
    filter_report_path: str,
    n_tic: int = 3000,
    n_no_tic: int = 3000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Sample frames from split, return embeddings, int labels, group names.
    """
    random.seed(seed)
    np.random.seed(seed)

    with open(filter_report_path) as f:
        filter_report = json.load(f)
    excluded_groups = set(filter_report["excluded_groups"])

    split_df  = pd.read_csv(split_csv)
    filenames = split_df["filename"].tolist()
    random.shuffle(filenames)

    tic_embs,    tic_labels,    tic_groups    = [], [], []
    no_tic_embs, no_tic_labels, no_tic_groups = [], [], []

    for filename in filenames:
        if len(tic_embs) >= n_tic and len(no_tic_embs) >= n_no_tic:
            break

        stem    = Path(filename).stem
        patient = stem.split("_")[0]
        pt_path = Path(embeddings_dir) / patient / f"{stem}.pt"

        if not pt_path.exists():
            continue

        frames = torch.load(pt_path, map_location="cpu", weights_only=False)
        random.shuffle(frames)

        for f in frames:
            if f["Group"] in excluded_groups:
                continue

            emb       = f["embedding"].numpy()
            tic_type  = f["Type"]
            group     = f["Group"]

            type_to_int = {int(k): v for k, v in label_config["type_to_int"].items()}
            int_label = type_to_int.get(tic_type, NO_TIC_INT)

            if int_label == NO_TIC_INT:
                if len(no_tic_embs) < n_no_tic:
                    no_tic_embs.append(emb)
                    no_tic_labels.append(NO_TIC_INT)
                    no_tic_groups.append("no-tic")
            else:
                if len(tic_embs) < n_tic:
                    tic_embs.append(emb)
                    tic_labels.append(int_label)
                    tic_groups.append(group if group != "-100" else "unknown")

    all_embs   = np.array(tic_embs + no_tic_embs)
    all_labels = np.array(tic_labels + no_tic_labels)
    all_groups = tic_groups + no_tic_groups

    print(f"[viz] Sampled {len(tic_embs)} tic frames, {len(no_tic_embs)} no-tic frames")
    print(f"[viz] Embedding shape: {all_embs.shape}")
    return all_embs, all_labels, all_groups


def _plot_2d(
    coords: np.ndarray,
    all_labels: np.ndarray,
    all_groups: list,
    title: str,
    output_path: Path,
    mode: str = "binary",
) -> None:
    """
    Plot 2D scatter — binary (tic vs no-tic) or per-group.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if mode == "binary":
        tic_mask    = all_labels != NO_TIC_INT
        no_tic_mask = ~tic_mask

        ax.scatter(
            coords[no_tic_mask, 0], coords[no_tic_mask, 1],
            c="#9e9b93", s=4, alpha=0.4, label=f"no-tic (n={no_tic_mask.sum():,})", rasterized=True
        )
        ax.scatter(
            coords[tic_mask, 0], coords[tic_mask, 1],
            c="#2563a8", s=6, alpha=0.6, label=f"tic (n={tic_mask.sum():,})", rasterized=True
        )
        ax.legend(markerscale=3, frameon=False, fontsize=10)

    elif mode == "groups":
        groups_present = sorted(set(all_groups))
        for group in groups_present:
            mask  = np.array(all_groups) == group
            color = COLORS.get(group, "#333333")
            size  = 3 if group == "no-tic" else 8
            alpha = 0.3 if group == "no-tic" else 0.7
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, s=size, alpha=alpha,
                label=f"{group} ({mask.sum():,})", rasterized=True
            )
        ax.legend(markerscale=2, frameon=False, fontsize=7,
                  loc="upper right", ncol=2)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved: {output_path}")


def run_visualization(
    n_tic: int = 3000,
    n_no_tic: int = 3000,
    tsne_perplexity: int = 50,
    seed: int = 42,
) -> None:
    output_dir = Path(paths_cfg["output_dir"])
    viz_dir    = output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    split_csv          = str(output_dir / "splits" / "file_split" / "train.csv")
    embeddings_dir     = paths_cfg["new_embeddings_dir"]
    filter_report_path = str(output_dir / "splits" / "file_split" / "filter_report.json")

    print(f"[viz] Sampling frames...")
    all_embs, all_labels, all_groups = _sample_frames(
        split_csv, embeddings_dir, filter_report_path,
        n_tic=n_tic, n_no_tic=n_no_tic, seed=seed
    )

    # -- PCA first (fast sanity check + t-SNE init) --
    print(f"[viz] Running PCA...")
    pca    = PCA(n_components=50, random_state=seed)
    embs_pca = pca.fit_transform(all_embs)
    print(f"[viz] PCA explained variance (top 10): {pca.explained_variance_ratio_[:10].round(3)}")

    pca_2d = PCA(n_components=2, random_state=seed)
    coords_pca = pca_2d.fit_transform(all_embs)

    _plot_2d(coords_pca, all_labels, all_groups,
             "PCA — WavLM embeddings (binary)",
             viz_dir / "pca_binary.png", mode="binary")
    _plot_2d(coords_pca, all_labels, all_groups,
             "PCA — WavLM embeddings (per group)",
             viz_dir / "pca_groups.png", mode="groups")

    # -- t-SNE --
    print(f"[viz] Running t-SNE (perplexity={tsne_perplexity})... this takes a few minutes")
    tsne = TSNE(
        n_components = 2,
        perplexity   = tsne_perplexity,
        max_iter     = 1000,
        init         = "pca",
        random_state = seed,
        n_jobs       = 4,
    )
    coords_tsne = tsne.fit_transform(embs_pca)

    _plot_2d(coords_tsne, all_labels, all_groups,
             "t-SNE — WavLM embeddings (binary)",
             viz_dir / "tsne_binary.png", mode="binary")
    _plot_2d(coords_tsne, all_labels, all_groups,
             "t-SNE — WavLM embeddings (per group)",
             viz_dir / "tsne_groups.png", mode="groups")

    # -- UMAP (optional) --
    try:
        import umap
        print(f"[viz] Running UMAP...")
        reducer     = umap.UMAP(n_components=2, random_state=seed, n_jobs=4)
        coords_umap = reducer.fit_transform(embs_pca)

        _plot_2d(coords_umap, all_labels, all_groups,
                 "UMAP — WavLM embeddings (binary)",
                 viz_dir / "umap_binary.png", mode="binary")
        _plot_2d(coords_umap, all_labels, all_groups,
                 "UMAP — WavLM embeddings (per group)",
                 viz_dir / "umap_groups.png", mode="groups")
    except ImportError:
        print(f"[viz] UMAP not installed, skipping. Install with: pip install umap-learn")

    print(f"\n[viz] All plots saved to: {viz_dir}")
    print(f"[viz] Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tic",      type=int, default=3000)
    parser.add_argument("--n-no-tic",   type=int, default=3000)
    parser.add_argument("--perplexity", type=int, default=50)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run_visualization(
        n_tic           = args.n_tic,
        n_no_tic        = args.n_no_tic,
        tsne_perplexity = args.perplexity,
        seed            = args.seed,
    )