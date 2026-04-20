"""
stages/s04b_filter.py
---------------------
Stage 04b: Filter

Determines which tic groups to exclude from training based on
two thresholds:
    - min_test_frames:  exclude group if test has < 100 frames
    - min_train_frames: exclude group if train has < 1000 frames

Excluded groups are masked in ALL splits — train, val and test.
No .pt files are modified. filter_report.json is read by
s05_dataset.py to skip excluded groups at dataset construction.

Inputs:
    configs/paths.yaml   -> output_dir
    configs/split.yaml   -> filter thresholds
    outputs/meta/group_counts.csv
    outputs/splits/file_split/train.csv etc.
    outputs/splits/patient_split/train.csv etc.

Outputs:
    outputs/splits/file_split/filter_report.json
    outputs/splits/patient_split/filter_report.json
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
import yaml
import logging

HOME_DIR = os.environ.get(
    "HOME_DIR",
    "/home/kzaveri1/codes/modular_pipline_package/tic_detection"
)
sys.path.insert(0, HOME_DIR)

with open(os.path.join(HOME_DIR, "configs", "paths.yaml"), "r") as f:
    paths_cfg = yaml.safe_load(f)

with open(os.path.join(HOME_DIR, "configs", "split.yaml"), "r") as f:
    split_cfg = yaml.safe_load(f)

def _compute_split_group_counts(
    split_dir: Path,
    group_counts_df: pd.DataFrame,
    all_groups: list[str],
) -> dict:
    """
    Sum group frame counts for each split (train, val, test)
    by joining split CSVs with group_counts.csv.

    Args:
        split_dir:       path to file_split/ or patient_split/
        group_counts_df: loaded group_counts.csv
        all_groups:      sorted list of all group names

    Returns:
        dict with keys train, val, test each mapping to
        {group_name: total_frame_count}
    """
    result = {}

    for split_name in ["train", "val", "test"]:
        split_csv = split_dir / f"{split_name}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"[s04b] Split CSV not found: {split_csv}")

        split_files = pd.read_csv(split_csv)["filename"].tolist()

        split_group_counts = group_counts_df[
            group_counts_df["filename"].isin(split_files)
        ][all_groups].sum()

        result[split_name] = {
            g: int(split_group_counts[g]) for g in all_groups
        }

    return result
def _compute_exclusions(
    split_counts: dict,
    all_groups: list[str],
) -> dict:
    """
    Determine which groups to exclude based on thresholds.
    A group is excluded from all splits if:
        - test frames < min_test_frames, OR
        - train frames < min_train_frames

    Args:
        split_counts: {train: {group: count}, val: {group: count}, test: {group: count}}
        all_groups:   sorted list of all group names

    Returns:
        dict with:
            excluded_groups:    list of group names to exclude
            included_groups:    list of group names to keep
            exclusion_reasons:  {group: reason string}
    """
    min_test_frames  = split_cfg["filter"]["min_test_frames"]
    min_train_frames = split_cfg["filter"]["min_train_frames"]

    excluded_groups   = []
    included_groups   = []
    exclusion_reasons = {}

    for group in all_groups:
        test_count  = split_counts["test"][group]
        train_count = split_counts["train"][group]

        if test_count < min_test_frames:
            excluded_groups.append(group)
            exclusion_reasons[group] = f"test={test_count} < {min_test_frames}"
        elif train_count < min_train_frames:
            excluded_groups.append(group)
            exclusion_reasons[group] = f"train={train_count} < {min_train_frames}"
        else:
            included_groups.append(group)

    return {
        "excluded_groups":   excluded_groups,
        "included_groups":   included_groups,
        "exclusion_reasons": exclusion_reasons,
    }
def run_filter() -> None:
    """
    Run the full filter stage.
    Computes exclusions for both file_split and patient_split.
    Saves filter_report.json to each split directory.
    """
    # -- setup logging --
    log_dir  = Path(paths_cfg["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "s04b_filter.log"

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
    log.info(f"[s04b] Loading group counts: {group_counts_path}")
    group_counts_df = pd.read_csv(group_counts_path)

    meta_cols  = ["filename", "ID", "Sess", "Phase"]
    all_groups = sorted([c for c in group_counts_df.columns if c not in meta_cols])
    log.info(f"[s04b] {len(all_groups)} tic groups found")

    splits_dir = Path(paths_cfg["output_dir"]) / "splits"

    # -- process both splits --
    for split_name in ["file_split", "patient_split"]:
        split_dir = splits_dir / split_name
        log.info(f"\n[s04b] ── {split_name} ──")

        # compute group counts per split
        split_counts = _compute_split_group_counts(
            split_dir, group_counts_df, all_groups
        )

        # compute exclusions
        exclusions = _compute_exclusions(split_counts, all_groups)

        # build full report

        report = {
            "split":             split_name,
            "thresholds": {
                "min_test_frames":  split_cfg["filter"]["min_test_frames"],
                "min_train_frames": split_cfg["filter"]["min_train_frames"],
            },
            "excluded_groups":   exclusions["excluded_groups"],
            "included_groups":   exclusions["included_groups"],
            "exclusion_reasons": exclusions["exclusion_reasons"],
            "split_counts":      split_counts,
        }

        # save report
        report_path = split_dir / "filter_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # log summary
        log.info(f"[s04b] Excluded groups ({len(exclusions['excluded_groups'])}):")
        for g, reason in exclusions["exclusion_reasons"].items():
            log.info(f"[s04b]   {g:<35} {reason}")

        log.info(f"[s04b] Included groups ({len(exclusions['included_groups'])}):")
        for g in exclusions["included_groups"]:
            train_c = split_counts["train"][g]
            val_c   = split_counts["val"][g]
            test_c  = split_counts["test"][g]
            log.info(f"[s04b]   {g:<35} train={train_c:>6,}  val={val_c:>6,}  test={test_c:>6,}")

        log.info(f"[s04b] Report saved to: {report_path}")

    log.info(f"\n[s04b] Done.")


if __name__ == "__main__":
    run_filter()