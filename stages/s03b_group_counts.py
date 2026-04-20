"""
stages/s03b_group_counts.py
---------------------------
Stage 03b: Group Counts

Scans all labeled .pt files in Embeddings_new/ and counts
frames per tic group per file. Excludes no-tic frames (-100).
Run once after s03_label.py completes.

Inputs:
    configs/paths.yaml      -> new_embeddings_dir, output_dir
    configs/tic_groups.csv
    outputs/meta/manifest.csv

Outputs:
    outputs/meta/group_counts.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
import yaml
import logging
from tqdm import tqdm

# ── home directory ───────────────────────────────────────────────────────────
HOME_DIR = os.environ.get(
    "HOME_DIR",
    "/home/kzaveri1/codes/modular_pipline_package/tic_detection"
)
sys.path.insert(0, HOME_DIR)

# ── configs ──────────────────────────────────────────────────────────────────
with open(os.path.join(HOME_DIR, "configs", "paths.yaml"), "r") as f:
    paths_cfg = yaml.safe_load(f)

# ── tic groups ───────────────────────────────────────────────────────────────
tic_groups_df = pd.read_csv(os.path.join(HOME_DIR, "configs", "tic_groups.csv"))
tic_groups    = dict(zip(tic_groups_df["Type"].astype(int), tic_groups_df["group"].astype(str)))
all_groups    = sorted(set(tic_groups.values()))
def _count_groups_in_file(pt_path: Path) -> dict:
    """
    Load a labeled .pt file and count frames per tic group.
    Excludes no-tic frames (Type == -100).

    Args:
        pt_path: path to labeled .pt file

    Returns:
        dict with group name -> frame count
    """
    frames = torch.load(pt_path, map_location="cpu")

    counts = {g: 0 for g in all_groups}

    for frame in frames:
        if frame["Type"] != -100:
            group = frame["Group"]
            if group in counts:
                counts[group] += 1

    return counts
def run_group_counts() -> pd.DataFrame:
    """
    Run the full group counts stage.
    Scans all labeled .pt files and saves group_counts.csv.

    Returns:
        group_counts_df — saved to outputs/meta/group_counts.csv
    """
    # -- setup logging --
    log_dir  = Path(paths_cfg["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "s03b_group_counts.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger(__name__)

    # -- load manifest --
    manifest_path = Path(paths_cfg["output_dir"]) / "meta" / "manifest.csv"
    manifest      = pd.read_csv(manifest_path)
    log.info(f"[s03b] {len(manifest)} files in manifest")

    # -- scan .pt files --
    embeddings_dir = Path(paths_cfg["new_embeddings_dir"])
    rows           = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Counting groups"):
        pt_path = (
            embeddings_dir
            / str(row["ID"])
            / f"{Path(row['filename']).stem}.pt"
        )

        if not pt_path.exists():
            log.info(f"[s03b] ⚠️  PT file not found, skipping: {pt_path}")
            continue

        try:
            counts = _count_groups_in_file(pt_path)
        except Exception as e:
            log.info(f"[s03b] ❌ Error reading {pt_path}: {e}")
            continue

        row_dict = {
            "filename": row["filename"],
            "ID":       row["ID"],
            "Sess":     row["Sess"],
            "Phase":    row["Phase"],
        }
        row_dict.update(counts)
        rows.append(row_dict)

    # -- save --
    group_counts_df   = pd.DataFrame(rows)
    group_counts_path = Path(paths_cfg["output_dir"]) / "meta" / "group_counts.csv"
    group_counts_df.to_csv(group_counts_path, index=False)

    log.info(f"[s03b] {len(group_counts_df)} files processed")
    log.info(f"[s03b] Group counts saved to: {group_counts_path}")

    # -- print summary --
    log.info(f"\n[s03b] ── Group Distribution ──")
    for group in all_groups:
        total = group_counts_df[group].sum()
        if total > 0:
            log.info(f"[s03b]   {group:<35} {total:>8,} frames")

    log.info(f"[s03b] Done.")
    return group_counts_df


if __name__ == "__main__":
    run_group_counts()