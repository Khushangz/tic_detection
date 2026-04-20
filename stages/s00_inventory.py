"""
stages/s00_inventory.py
-----------------------
Stage 00: Inventory
"""

import os
import sys
from pathlib import Path
import pandas as pd
import yaml

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


# ---------------------------------------------------------------------------
# Validate tic CSV
# ---------------------------------------------------------------------------

def _validate_tic_csv(df: pd.DataFrame) -> None:
    """
    Validate the tic annotation CSV before any processing.
    Raises ValueError immediately with a clear message on any issue.
    """

    # -- required columns --
    required = {"ID", "Sess", "Phase", "Type", "StartTime", "EndTime",
                "Duration", "Audio_duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[s00] tic_csv missing required columns: {missing}")

    # -- no NaN in critical columns --
    for col in ["ID", "Sess", "Phase", "Type", "StartTime", "EndTime"]:
        n_null = df[col].isna().sum()
        if n_null > 0:
            raise ValueError(f"[s00] {n_null} NaN values found in column '{col}'")

    # -- no negative StartTime --
    if (df["StartTime"] < 0).any():
        raise ValueError("[s00] Negative StartTime values found in tic_csv.")

    # -- EndTime must be greater than StartTime --
    bad = df[df["EndTime"] <= df["StartTime"]]
    if len(bad) > 0:
        raise ValueError(
            f"[s00] {len(bad)} rows where EndTime <= StartTime:\n"
            f"{bad[['ID', 'Sess', 'Phase', 'StartTime', 'EndTime']].head()}"
        )

    print(f"[s00] CSV validation passed. {len(df)} rows, {df.columns.tolist()}")

def _build_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per unique ID+Sess+Phase combination.
    Adds filename column if not already present.
    """
    if "filename" not in df.columns:
        df = df.copy()
        df["filename"] = (
            df["ID"].astype(str)
            + "_V" + df["Sess"].astype(str)
            + "_" + df["Phase"].astype(str)
            + ".wav"
        )

    manifest = (
        df.groupby(["ID", "Sess", "Phase", "filename", "Audio_duration"], as_index=False)
        .agg(n_tics=("Type", "count"), n_tic_types=("Type", "nunique"))
        .sort_values(["ID", "Sess", "Phase"])
        .reset_index(drop=True)
    )

    return manifest

def _check_pt_files(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    For each file in manifest check whether its .pt file exists.
    Tries primary name first then _Front fallback.
    Adds: pt_path (str), pt_found (bool)
    """
    embeddings_dir = Path(paths_cfg["embeddings_dir"])
    pt_paths = []
    pt_found = []

    for _, row in manifest.iterrows():
        primary = (
            embeddings_dir
            / str(row["ID"])
            / f"{row['ID']}_V{row['Sess']}_{row['Phase']}.pt"
        )
        fallback = (
            embeddings_dir
            / str(row["ID"])
            / f"{row['ID']}_V{row['Sess']}_{row['Phase']}_Front.pt"
        )

        if primary.exists():
            pt_paths.append(str(primary))
            pt_found.append(True)
        elif fallback.exists():
            pt_paths.append(str(fallback))
            pt_found.append(True)
        else:
            pt_paths.append("")
            pt_found.append(False)

    manifest = manifest.copy()
    manifest["pt_path"] = pt_paths
    manifest["pt_found"] = pt_found
    return manifest

def _build_summary(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Build dataset-level statistics as a two-column metric/value CSV.
    Everything derived dynamically from the data columns.
    """
    rows = []

    # -- file level --
    rows.append(("total_files_in_csv",            len(manifest)))
    rows.append(("total_pt_files_found",           int(manifest["pt_found"].sum())))
    rows.append(("total_pt_files_missing",         int((~manifest["pt_found"]).sum())))
    rows.append(("total_files_below_min_duration", int(manifest["below_min_duration"].sum())))

    # -- tic level --
    rows.append(("total_tic_rows",                 len(df)))
    rows.append(("total_unique_tic_types",         df["Type"].nunique()))

    # -- patient / session / phase --
    rows.append(("total_unique_patients",          df["ID"].nunique()))
    rows.append(("total_unique_sessions",          df["Sess"].nunique()))
    rows.append(("total_unique_phases",            df["Phase"].nunique()))

    # -- per patient --
    for patient_id, grp in manifest.groupby("ID"):
        rows.append((f"files_patient_{patient_id}", len(grp)))

    # -- per session --
    for sess, grp in manifest.groupby("Sess"):
        rows.append((f"files_session_{sess}", len(grp)))

    # -- per phase --
    for phase, grp in manifest.groupby("Phase"):
        rows.append((f"files_phase_{phase}", len(grp)))

    # -- per tic type --
    for tic_type, grp in df.groupby("Type"):
        rows.append((f"tic_rows_type_{tic_type}", len(grp)))

    return pd.DataFrame(rows, columns=["metric", "value"])

def run_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full inventory stage.
    Reads paths and audio config from module-level configs.

    Returns:
        (manifest_df, summary_df) — both saved to outputs/meta/
    """
    # -- setup output dir --
    output_dir = Path(paths_cfg["output_dir"])
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # -- load tic csv --
    print(f"[s00] Loading tic CSV: {paths_cfg['tic_csv']}")
    df = pd.read_csv(paths_cfg["tic_csv"])

    # -- validate --
    _validate_tic_csv(df)

    # -- build manifest --
    print(f"[s00] Building manifest...")
    manifest = _build_manifest(df)
    print(f"[s00] {len(manifest)} unique files found")

    # -- check pt files --
    print(f"[s00] Checking .pt files...")
    manifest = _check_pt_files(manifest)

    # -- flag short files --
    min_dur = audio_cfg["min_duration_s"]
    manifest["below_min_duration"] = manifest["Audio_duration"] < min_dur

    # -- build summary --
    print(f"[s00] Building summary...")
    summary = _build_summary(df, manifest)

    # -- save --
    manifest_path = meta_dir / "manifest.csv"
    summary_path  = meta_dir / "inventory_summary.csv"

    manifest.to_csv(manifest_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"[s00] Manifest saved to: {manifest_path}")
    print(f"[s00] Summary saved to:  {summary_path}")

    # -- print top level stats --
    print(f"\n[s00] ── Inventory Summary ──")
    top_keys = [
        "total_files_in_csv",
        "total_pt_files_found",
        "total_pt_files_missing",
        "total_files_below_min_duration",
        "total_tic_rows",
        "total_unique_tic_types",
        "total_unique_patients",
        "total_unique_sessions",
        "total_unique_phases",
    ]
    for key in top_keys:
        row = summary[summary["metric"] == key]
        if not row.empty:
            print(f"[s00]   {key:<40} {row['value'].iloc[0]}")

    print(f"\n[s00] Done.")
    return manifest, summary


if __name__ == "__main__":
    run_inventory()