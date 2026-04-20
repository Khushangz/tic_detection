"""
utils/labels.py
---------------
Pure functions for tic label processing.
No file I/O here — all path handling lives in stages/s03_label.py.

Pipeline role:
    1. gap_fill        — insert no-tic rows between tic intervals
    2. segment_20ms    — expand every row into 20ms frames
    3. build_label_config — emit label schema for downstream stages
"""

from decimal import Decimal, getcontext
from collections import defaultdict

import pandas as pd

# Decimal precision for 20ms segmentation — avoids float drift
getcontext().prec = 10

NO_TIC_LABEL: int = -100
FRAME_DURATION: Decimal = Decimal("0.02")
EPSILON: Decimal = Decimal("1e-8")


# ---------------------------------------------------------------------------
# 1. Gap fill
# ---------------------------------------------------------------------------

def gap_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert no-tic rows (-100) to cover gaps between tic intervals.

    Expects columns: ID, Sess, Phase, Type, StartTime, EndTime,
                     Duration, Audio_duration, filename

    Files are processed independently. Within each file, rows are
    sorted by StartTime. A no-tic row is inserted wherever the previous
    tic ends before the next tic begins.

    Returns a new DataFrame with no-tic rows inserted, sorted by
    filename then StartTime.
    """
    rows = []
    df = df.sort_values(by=["filename", "StartTime"]).reset_index(drop=True)

    current_file = None
    cursor = 0.0  # tracks end of last interval within a file

    for _, row in df.iterrows():
        if row["filename"] != current_file:
            current_file = row["filename"]
            cursor = 0.0

        if cursor < row["StartTime"]:
            gap = row.copy()
            gap["Type"] = NO_TIC_LABEL
            gap["StartTime"] = cursor
            gap["EndTime"] = row["StartTime"]
            gap["Duration"] = round(row["StartTime"] - cursor, 6)
            rows.append(gap)

        rows.append(row)
        cursor = row["EndTime"]

    result = pd.DataFrame(rows).reset_index(drop=True)
    return result.sort_values(by=["filename", "StartTime"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Segment into 20ms frames
# ---------------------------------------------------------------------------

def segment_20ms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand every row into consecutive 20ms frames.

    Uses Decimal arithmetic to avoid floating point drift accumulating
    over thousands of frames in a long audio file.

    Each output row represents one 20ms frame and inherits all metadata
    columns from its source row (ID, Sess, Phase, Type, filename, etc).
    StartTime and EndTime are replaced with the frame's own boundaries.
    Duration is always 0.02.

    Returns one row per 20ms frame, sorted by filename then StartTime.
    """
    rows = []

    for _, row in df.iterrows():
        start = Decimal(str(row["StartTime"]))
        end = Decimal(str(row["EndTime"]))

        t = start
        while t + FRAME_DURATION <= end + EPSILON:
            frame = row.copy()
            frame["StartTime"] = float(t)
            frame["EndTime"] = float(t + FRAME_DURATION)
            frame["Duration"] = float(FRAME_DURATION)
            rows.append(frame)
            t += FRAME_DURATION

    result = pd.DataFrame(rows).reset_index(drop=True)
    return result.sort_values(by=["filename", "StartTime"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Attach group IDs
# ---------------------------------------------------------------------------

def attach_groups(df: pd.DataFrame, tic_groups: dict[int, str]) -> pd.DataFrame:
    """
    Add a Group column by mapping Type → group name via tic_groups dict.

    No-tic frames (Type == NO_TIC_LABEL) get Group = NO_TIC_LABEL as a string.
    Any Type not found in tic_groups raises a ValueError immediately — 
    never silently produces NaN.

    Args:
        df:          segmented DataFrame with a Type column
        tic_groups:  {tic_type_int: group_name_str}

    Returns:
        df with a new Group column added
    """
    unknown = (
        set(df["Type"].unique())
        - set(tic_groups.keys())
        - {NO_TIC_LABEL}
    )
    if unknown:
        raise ValueError(
            f"Types in data not found in tic_groups config: {unknown}. "
            f"Add them to configs/tic_groups.csv before running."
        )

    df = df.copy()
    df["Group"] = df["Type"].apply(
        lambda t: str(NO_TIC_LABEL) if t == NO_TIC_LABEL else tic_groups[t]
    )
    return df


# ---------------------------------------------------------------------------
# 4. Load tic groups
# ---------------------------------------------------------------------------

def load_tic_groups(tic_groups_csv: str) -> dict[int, str]:
    """
    Load configs/tic_groups.csv into {Type_int: group_name_str}.

    Expected CSV format:
        Type,group
        1009,Throat Clearing
        1010,Sniffing
        ...

    Returns:
        dict mapping integer Type ID to group name string
    """
    df = pd.read_csv(tic_groups_csv)

    required = {"Type", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tic_groups.csv missing columns: {missing}")

    return dict(zip(df["Type"].astype(int), df["group"].astype(str)))


# ---------------------------------------------------------------------------
# 5. Build label config
# ---------------------------------------------------------------------------

def build_label_config(frames_df: pd.DataFrame, tic_groups: dict[int, str]) -> dict:
    """
    Emit the label schema used by all downstream stages.

    Computes integer encodings for each tic Type, class counts,
    and group membership — all derived from the fully segmented
    frames DataFrame. Saved once at end of s03, never re-inferred.

    Returns a dict with:
        no_tic_label        int
        tic_types           list[int]   all Type IDs excluding no-tic
        type_to_int         dict        Type int → class index int
        int_to_type         dict        class index int → Type int
        type_to_group       dict        Type int → group name str
        group_names         list[str]   unique group names
        class_counts        dict        Type int → frame count
        num_classes         int         total number of classes inc no-tic
    """
    all_types = sorted(frames_df["Type"].unique().tolist())
    tic_types = [t for t in all_types if t != NO_TIC_LABEL]

    # Integer encoding: 0..N-1 for tic types, N for no-tic
    type_to_int = {t: i for i, t in enumerate(tic_types)}
    type_to_int[NO_TIC_LABEL] = len(tic_types)

    int_to_type = {v: k for k, v in type_to_int.items()}

    class_counts = frames_df["Type"].value_counts().to_dict()
    # ensure all types present even if count is 0
    for t in all_types:
        class_counts.setdefault(t, 0)

    group_names = sorted(set(tic_groups.values()))

    return {
        "no_tic_label": NO_TIC_LABEL,
        "tic_types": tic_types,
        "type_to_int": type_to_int,
        "int_to_type": int_to_type,
        "type_to_group": {t: tic_groups.get(t, str(NO_TIC_LABEL)) for t in tic_types},
        "group_names": group_names,
        "class_counts": class_counts,
        "num_classes": len(all_types),
    }