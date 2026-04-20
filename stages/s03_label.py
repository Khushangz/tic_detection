"""
stages/s03_label.py
-------------------
Stage 03: Label

For each .pt file in Embeddings_new/, assigns a Type and Group
to every 20ms frame by checking if the frame's start_time_s
falls within any tic interval in filtered_ticList_metadata.csv.

Overwrites each .pt file in place with Type and Group added
to every frame dict.

Inputs:
    configs/paths.yaml    -> tic_csv, new_embeddings_dir
    configs/tic_groups.csv
    outputs/meta/manifest.csv

Outputs:
    Embeddings_new/{ID}/{filename}.pt  (overwritten with labels)
    outputs/meta/frames_summary.csv    one row per file with label counts
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
import yaml
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

with open(os.path.join(HOME_DIR, "configs", "audio.yaml"), "r") as f:
    audio_cfg = yaml.safe_load(f)

def _load_tic_groups() -> dict[int, str]:
    """
    Load configs/tic_groups.csv into {Type_int: group_name_str}.
    Raises if any Type is missing or file is malformed.
    """
    tic_groups_path = os.path.join(HOME_DIR, "configs", "tic_groups.csv")
    df = pd.read_csv(tic_groups_path)

    required = {"Type", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[s03] tic_groups.csv missing columns: {missing}")

    return dict(zip(df["Type"].astype(int), df["group"].astype(str)))

def _load_tic_intervals(tic_csv: str) -> dict[str, list[dict]]:
    """
    Load filtered_ticList_metadata.csv and build a per-file lookup
    of tic intervals.

    Returns:
        dict mapping filename -> list of dicts:
            [{"Type": int, "StartTime": float, "EndTime": float}, ...]
        Sorted by StartTime per file.
    """
    df = pd.read_csv(tic_csv)

    # build filename if not present
    if "filename" not in df.columns:
        df["filename"] = (
            df["ID"].astype(str)
            + "_V" + df["Sess"].astype(str)
            + "_" + df["Phase"].astype(str)
            + ".wav"
        )

    intervals = {}
    for filename, group in df.groupby("filename"):
        group = group.sort_values("StartTime")
        intervals[filename] = [
            {
                "Type":      int(row["Type"]),
                "StartTime": float(row["StartTime"]),
                "EndTime":   float(row["EndTime"]),
            }
            for _, row in group.iterrows()
        ]

    return intervals

def _assign_label(
    frame_start: float,
    tic_intervals: list[dict],
) -> tuple[int, str]:
    """
    Assign a Type and Group to a single 20ms frame.

    A frame belongs to a tic if frame_start >= tic_start
    and frame_start < tic_end.

    Args:
        frame_start:   start_time_s of the frame
        tic_intervals: sorted list of tic dicts for this file

    Returns:
        (Type, Group) — (-100, "-100") if no tic covers this frame
    """
    for tic in tic_intervals:
        if frame_start >= tic["StartTime"] and frame_start < tic["EndTime"]:
            return tic["Type"], tic_groups[tic["Type"]]

    return -100, "-100"

def _label_pt_file(
    pt_path: Path,
    tic_intervals: list[dict],
) -> dict:
    """
    Load a .pt file, assign Type and Group to every frame,
    overwrite the file in place.

    Args:
        pt_path:       path to the .pt file in Embeddings_new/
        tic_intervals: sorted tic intervals for this file

    Returns:
        summary dict with label counts for frames_summary.csv
    """
    frames = torch.load(pt_path, map_location="cpu")

    tic_count    = 0
    no_tic_count = 0
    type_counts  = {}

    for frame in frames:
        frame_start         = frame["start_time_s"]
        tic_type, tic_group = _assign_label(frame_start, tic_intervals)

        frame["Type"]  = tic_type
        frame["Group"] = tic_group

        if tic_type == -100:
            no_tic_count += 1
        else:
            tic_count += 1
            type_counts[tic_type] = type_counts.get(tic_type, 0) + 1

    torch.save(frames, pt_path)

    return {
        "pt_path":       str(pt_path),
        "total_frames":  len(frames),
        "tic_frames":    tic_count,
        "no_tic_frames": no_tic_count,
        "type_counts":   type_counts,
    }

def run_label() -> pd.DataFrame:
    """
    Run the full label stage.
    Loads tic intervals, assigns Type and Group to every frame
    in every .pt file in Embeddings_new/, overwrites in place.
    Saves frames_summary.csv with label counts per file.

    Returns:
        summary_df — saved to outputs/meta/frames_summary.csv
    """
    import logging

    # -- setup logging --
    log_dir  = Path(paths_cfg["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "s03_label.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger(__name__)

    # -- load tic groups --
    global tic_groups
    tic_groups = _load_tic_groups()
    log.info(f"[s03] {len(tic_groups)} tic types loaded from tic_groups.csv")

    # -- load tic intervals --
    tic_csv = paths_cfg["tic_csv"]
    log.info(f"[s03] Loading tic intervals from: {tic_csv}")
    all_intervals = _load_tic_intervals(tic_csv)
    log.info(f"[s03] {len(all_intervals)} files with tic intervals")

    # -- load manifest --
    manifest_path = Path(paths_cfg["output_dir"]) / "meta" / "manifest.csv"
    manifest      = pd.read_csv(manifest_path)
    log.info(f"[s03] {len(manifest)} files in manifest")

    # -- process each file --
    embeddings_dir = Path(paths_cfg["new_embeddings_dir"])
    summaries      = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Labeling"):
        pt_path = (
            embeddings_dir
            / str(row["ID"])
            / f"{Path(row['filename']).stem}.pt"
        )

        if not pt_path.exists():
            log.info(f"[s03] ⚠️  PT file not found, skipping: {pt_path}")
            continue

        filename = row["filename"]
        if filename not in all_intervals:
            log.info(f"[s03] ⚠️  No tic intervals found for: {filename}, all frames → -100")
            tic_intervals = []
        else:
            tic_intervals = all_intervals[filename]

        try:
            summary = _label_pt_file(pt_path, tic_intervals)
            summary["ID"]       = row["ID"]
            summary["Sess"]     = row["Sess"]
            summary["Phase"]    = row["Phase"]
            summary["filename"] = filename
            summaries.append(summary)
            log.info(
                f"[s03] ✅ {filename} | "
                f"tic={summary['tic_frames']} "
                f"no_tic={summary['no_tic_frames']} "
                f"total={summary['total_frames']}"
            )
        except Exception as e:
            log.info(f"[s03] ❌ Error labeling {pt_path}: {e}")
            continue

    # -- save summary --
    summary_df    = pd.DataFrame(summaries)
    summary_path  = Path(paths_cfg["output_dir"]) / "meta" / "frames_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    log.info(f"\n[s03] ── Label Summary ──")
    log.info(f"[s03] Total files labeled:  {len(summary_df)}")
    log.info(f"[s03] Total frames:         {summary_df['total_frames'].sum():,}")
    log.info(f"[s03] Total tic frames:     {summary_df['tic_frames'].sum():,}")
    log.info(f"[s03] Total no-tic frames:  {summary_df['no_tic_frames'].sum():,}")
    log.info(f"[s03] Summary saved to:     {summary_path}")
    log.info(f"[s03] Done.")
    # -- save label config --
    import json
    tic_types   = sorted(tic_groups.keys())
    type_to_int = {t: i for i, t in enumerate(tic_types)}
    type_to_int[-100] = len(tic_types)
    int_to_type = {v: k for k, v in type_to_int.items()}
    label_cfg = {
        "no_tic_label":  -100,
        "tic_types":     tic_types,
        "type_to_int":   type_to_int,
        "int_to_type":   int_to_type,
        "type_to_group": {t: tic_groups[t] for t in tic_types},
        "group_names":   sorted(set(tic_groups.values())),
        "num_classes":   len(tic_types) + 1,
    }
    label_cfg_path = Path(HOME_DIR) / "configs" / "label_config.json"
    with open(label_cfg_path, "w") as f:
        json.dump(label_cfg, f, indent=2)
    log.info(f"[s03] Label config saved to: {label_cfg_path}")

    return summary_df


if __name__ == "__main__":
    run_label()