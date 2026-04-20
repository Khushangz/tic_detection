"""
stages/s04_split.py
-------------------
Stage 04: Split

Runs JSD-optimized greedy split search to assign files to
train/val/test. Produces two splits:
    - file_split:    files are atomic, patients can repeat
    - patient_split: all files of a patient go to same split

Scoring uses three penalties:
    1. Train must have >= min_train_pct of total tic frames
    2. JSD distance between train/val/test distributions minimized
    3. Val must have more frames than test

Inputs:
    configs/paths.yaml   -> output_dir
    configs/split.yaml   -> split parameters
    outputs/meta/group_counts.csv
    outputs/meta/manifest.csv

Outputs:
    outputs/splits/file_split/train.csv, val.csv, test.csv
    outputs/splits/patient_split/train.csv, val.csv, test.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import yaml
import logging
import random
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
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

with open(os.path.join(HOME_DIR, "configs", "split.yaml"), "r") as f:
    split_cfg = yaml.safe_load(f)
def _get_distribution(
    file_list: list[str],
    file_analysis: dict,
    all_groups: list[str],
) -> tuple[dict, dict, int]:
    """
    Compute tic group distribution for a set of files.

    Args:
        file_list:     list of filenames
        file_analysis: {filename: {group: count}}
        all_groups:    sorted list of all group names

    Returns:
        counts:       {group: total frame count}
        distribution: {group: fraction of total}
        total:        total tic frames
    """
    counts = defaultdict(int)
    for fname in file_list:
        for group, count in file_analysis[fname].items():
            counts[group] += count
    for group in all_groups:
        counts.setdefault(group, 0)

    total = sum(counts.values())
    if total == 0:
        distribution = {g: 0.0 for g in all_groups}
    else:
        distribution = {g: counts[g] / total for g in all_groups}

    return dict(counts), distribution, total
def _score_split(
    train_files: list[str],
    val_files: list[str],
    test_files: list[str],
    file_analysis: dict,
    all_groups: list[str],
    total_tic_frames: int,
) -> tuple[float, dict]:
    """
    Score a split using three penalties.

    Penalty 1: Train must have >= min_train_pct of total tic frames
    Penalty 2: JSD distance between train/val/test distributions
    Penalty 3: Val must have more frames than test

    Returns:
        total_score: float (lower is better)
        details:     dict with breakdown of scores and distributions
    """
    min_train_pct = split_cfg["min_train_pct"]

    train_counts, train_dist, train_frames = _get_distribution(train_files, file_analysis, all_groups)
    val_counts,   val_dist,   val_frames   = _get_distribution(val_files,   file_analysis, all_groups)
    test_counts,  test_dist,  test_frames  = _get_distribution(test_files,  file_analysis, all_groups)

    # -- penalty 1: train size --
    train_pct     = train_frames / total_tic_frames if total_tic_frames > 0 else 0
    train_penalty = 0
    if train_pct < min_train_pct:
        train_penalty = (min_train_pct - train_pct) * 10000

    # -- penalty 2: JSD distribution distance --
    def jsd(d1, d2):
        v1 = [d1.get(g, 0) + 1e-10 for g in all_groups]
        v2 = [d2.get(g, 0) + 1e-10 for g in all_groups]
        return jensenshannon(v1, v2)

    train_val_dist  = jsd(train_dist, val_dist)
    train_test_dist = jsd(train_dist, test_dist)
    val_test_dist   = jsd(val_dist,   test_dist)
    avg_distance    = (train_val_dist + train_test_dist + val_test_dist) / 3
    distribution_penalty = avg_distance * 1000

    # -- penalty 3: val must have more frames than test --
    val_test_size_penalty = 0
    if val_frames <= test_frames:
        val_test_size_penalty = (test_frames - val_frames + 1) * 100

    total_score = train_penalty + distribution_penalty + val_test_size_penalty

    return total_score, {
        "train_pct":              train_pct,
        "train_frames":           train_frames,
        "val_frames":             val_frames,
        "test_frames":            test_frames,
        "avg_jsd":                avg_distance,
        "train_penalty":          train_penalty,
        "distribution_penalty":   distribution_penalty,
        "val_test_size_penalty":  val_test_size_penalty,
        "total_score":            total_score,
        "train_counts":           train_counts,
        "val_counts":             val_counts,
        "test_counts":            test_counts,
        "train_dist":             train_dist,
        "val_dist":               val_dist,
        "test_dist":              test_dist,
    }
def _greedy_split_search(
    files: list[str],
    file_analysis: dict,
    all_groups: list[str],
    total_tic_frames: int,
) -> tuple[list, list, list, dict]:
    """
    Greedy swap search to find optimal train/val/test split.
    At each iteration tries swapping files between splits.
    Keeps the swap if it improves the score.
    Stops when no improvement found or max_iterations reached.

    Args:
        files:            list of all filenames to split
        file_analysis:    {filename: {group: count}}
        all_groups:       sorted list of all group names
        total_tic_frames: total tic frames across all files

    Returns:
        train_files, val_files, test_files, best_details
    """
    min_train_files = split_cfg["min_train_files"]
    max_iterations  = split_cfg["max_iterations"]
    seed            = split_cfg["seed"]

    random.seed(seed)
    n_files = len(files)

    print(f"[s04] Greedy search | {n_files} files | "
          f"min_train_pct={split_cfg['min_train_pct']*100:.0f}%")

    # -- initial split --
    train_size = max(min_train_files, int(n_files * 0.75))
    remaining  = n_files - train_size
    val_size   = max(1, remaining - remaining // 3)

    shuffled     = files.copy()
    random.shuffle(shuffled)
    train_files  = shuffled[:train_size]
    val_files    = shuffled[train_size:train_size + val_size]
    test_files   = shuffled[train_size + val_size:]

    best_score, best_details = _score_split(
        train_files, val_files, test_files,
        file_analysis, all_groups, total_tic_frames
    )
    print(f"[s04] Initial score: {best_score:.2f} | "
          f"train: {best_details['train_pct']*100:.1f}%")

    # -- greedy swap --
    improved  = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved  = False
        iteration += 1

        for i, s1 in enumerate([train_files, val_files, test_files]):
            for j, s2 in enumerate([train_files, val_files, test_files]):
                if i >= j:
                    continue
                if i == 0 and len(s1) <= min_train_files:
                    continue

                for f1 in s1[:]:
                    for f2 in s2[:]:
                        new_splits = [
                            train_files.copy(),
                            val_files.copy(),
                            test_files.copy()
                        ]
                        new_splits[i].remove(f1)
                        new_splits[i].append(f2)
                        new_splits[j].remove(f2)
                        new_splits[j].append(f1)

                        new_score, new_details = _score_split(
                            new_splits[0], new_splits[1], new_splits[2],
                            file_analysis, all_groups, total_tic_frames
                        )

                        if new_score < best_score:
                            best_score   = new_score
                            best_details = new_details
                            train_files, val_files, test_files = new_splits
                            improved = True
                            print(f"[s04] Iter {iteration}: "
                                  f"score={best_score:.2f} | "
                                  f"train={best_details['train_pct']*100:.1f}%")
                            break
                    if improved: break
                if improved: break
            if improved: break

    print(f"[s04] Converged | score={best_score:.2f} | "
          f"train={best_details['train_pct']*100:.1f}%")

    return train_files, val_files, test_files, best_details
def _save_split(
    train_files: list[str],
    val_files: list[str],
    test_files: list[str],
    manifest: pd.DataFrame,
    split_dir: Path,
    details: dict,
    all_groups: list[str],
) -> None:
    """
    Save train/val/test CSVs and a split summary JSON.

    Args:
        train_files: list of filenames assigned to train
        val_files:   list of filenames assigned to val
        test_files:  list of filenames assigned to test
        manifest:    full manifest DataFrame
        split_dir:   output directory for this split
        details:     score details from _score_split
        all_groups:  sorted list of all group names
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    # -- save CSVs --
    for split_name, split_files in [
        ("train", train_files),
        ("val",   val_files),
        ("test",  test_files),
    ]:
        split_df = manifest[manifest["filename"].isin(split_files)].copy()
        split_df.to_csv(split_dir / f"{split_name}.csv", index=False)

    # -- save summary JSON --
    summary = {
        "train_files":   train_files,
        "val_files":     val_files,
        "test_files":    test_files,
        "train_frames":  details["train_frames"],
        "val_frames":    details["val_frames"],
        "test_frames":   details["test_frames"],
        "train_pct":     round(details["train_pct"] * 100, 2),
        "avg_jsd":       round(details["avg_jsd"], 6),
        "total_score":   round(details["total_score"], 4),
        "group_distribution": {
            g: {
                "train": details["train_counts"].get(g, 0),
                "val":   details["val_counts"].get(g, 0),
                "test":  details["test_counts"].get(g, 0),
            }
            for g in all_groups
        }
    }

    with open(split_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # -- print table --
    total_tic = details["train_frames"] + details["val_frames"] + details["test_frames"]
    print(f"\n[s04] ── Split Results ──")
    print(f"  Train : {len(train_files):3d} files | "
          f"{details['train_frames']:6d} frames "
          f"({details['train_pct']*100:.1f}%)")
    print(f"  Val   : {len(val_files):3d} files | "
          f"{details['val_frames']:6d} frames "
          f"({details['val_frames']/total_tic*100:.1f}%)")
    print(f"  Test  : {len(test_files):3d} files | "
          f"{details['test_frames']:6d} frames "
          f"({details['test_frames']/total_tic*100:.1f}%)")
    print(f"  Avg JSD:     {details['avg_jsd']:.4f}")
    print(f"  Total score: {details['total_score']:.4f}")
def run_split() -> None:
    """
    Run the full split stage.
    Produces two splits:
        - file_split:    files are atomic, patients can repeat
        - patient_split: all files of a patient go to same split
    """
    # -- setup logging --
    log_dir  = Path(paths_cfg["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "s04_split.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger(__name__)

    # -- load group counts --
    group_counts_path = Path(paths_cfg["output_dir"]) / "meta" / "group_counts.csv"
    log.info(f"[s04] Loading group counts: {group_counts_path}")
    group_counts_df = pd.read_csv(group_counts_path)

    # -- load manifest --
    manifest_path = Path(paths_cfg["output_dir"]) / "meta" / "manifest.csv"
    manifest      = pd.read_csv(manifest_path)

    # -- identify all groups --
    meta_cols  = ["filename", "ID", "Sess", "Phase"]
    all_groups = sorted([c for c in group_counts_df.columns if c not in meta_cols])
    log.info(f"[s04] {len(all_groups)} tic groups found")

    # -- build file_analysis: {filename: {group: count}} --
    file_analysis = {}
    for _, row in group_counts_df.iterrows():
        file_analysis[row["filename"]] = {
            g: int(row[g]) for g in all_groups
        }

    # -- total tic frames --
    total_tic_frames = sum(
        sum(counts.values())
        for counts in file_analysis.values()
    )
    log.info(f"[s04] Total tic frames: {total_tic_frames:,}")

    all_files = sorted(group_counts_df["filename"].tolist())

    # ------------------------------------------------------------------ #
    # 1. File split — use default split_cfg
    # ------------------------------------------------------------------ #
    log.info(f"\n[s04] ── File Split ──")
    train_files, val_files, test_files, details = _greedy_split_search(
        all_files, file_analysis, all_groups, total_tic_frames
    )
    file_split_dir = Path(paths_cfg["output_dir"]) / "splits" / "file_split"
    _save_split(
        train_files, val_files, test_files,
        manifest, file_split_dir, details, all_groups
    )
    log.info(f"[s04] File split saved to: {file_split_dir}")

    # ------------------------------------------------------------------ #
    # 2. Patient split — use patient_split config
    # ------------------------------------------------------------------ #
    log.info(f"\n[s04] ── Patient Split ──")

    # -- temporarily override split_cfg with patient_split params --
    patient_cfg = split_cfg.get("patient_split", split_cfg)
    original_min_train_pct   = split_cfg["min_train_pct"]
    original_min_train_files = split_cfg["min_train_files"]
    original_max_iterations  = split_cfg["max_iterations"]

    split_cfg["min_train_pct"]   = patient_cfg["min_train_pct"]
    split_cfg["min_train_files"] = patient_cfg["min_train_files"]
    split_cfg["max_iterations"]  = patient_cfg["max_iterations"]

    # -- group files by patient --
    patient_to_files = defaultdict(list)
    for fname in all_files:
        patient_id = group_counts_df[
            group_counts_df["filename"] == fname
        ]["ID"].iloc[0]
        patient_to_files[patient_id].append(fname)

    # -- build patient-level analysis --
    patient_analysis = {}
    for patient_id, p_files in patient_to_files.items():
        counts = defaultdict(int)
        for fname in p_files:
            for g, c in file_analysis[fname].items():
                counts[g] += c
        patient_analysis[patient_id] = dict(counts)

    all_patients = sorted(patient_to_files.keys())
    total_patient_tic_frames = sum(
        sum(counts.values())
        for counts in patient_analysis.values()
    )

    # -- run greedy search at patient level --
    train_patients, val_patients, test_patients, p_details = _greedy_split_search(
        all_patients, patient_analysis, all_groups, total_patient_tic_frames
    )

    # -- restore original split_cfg --
    split_cfg["min_train_pct"]   = original_min_train_pct
    split_cfg["min_train_files"] = original_min_train_files
    split_cfg["max_iterations"]  = original_max_iterations

    # -- expand patients back to files --
    train_files_p = [f for p in train_patients for f in patient_to_files[p]]
    val_files_p   = [f for p in val_patients   for f in patient_to_files[p]]
    test_files_p  = [f for p in test_patients  for f in patient_to_files[p]]

    # -- rescore at file level for accurate frame counts --
    _, p_details_files = _score_split(
        train_files_p, val_files_p, test_files_p,
        file_analysis, all_groups, total_tic_frames
    )

    patient_split_dir = Path(paths_cfg["output_dir"]) / "splits" / "patient_split"
    _save_split(
        train_files_p, val_files_p, test_files_p,
        manifest, patient_split_dir, p_details_files, all_groups
    )
    log.info(f"[s04] Patient split saved to: {patient_split_dir}")
    log.info(f"\n[s04] Done.")


if __name__ == "__main__":
    run_split()


if __name__ == "__main__":
    run_split()