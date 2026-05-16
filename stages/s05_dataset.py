"""
stages/s05_dataset.py
---------------------
Stage 05: Dataset

PyTorch Dataset class that reads labeled .pt files from
Embeddings_new/ and serves sequences to the BiLSTM.

Caches all .pt files in memory on init for fast access.
Skips frames whose Group is in filter_report excluded_groups.
Sequences never cross file boundaries.
Optionally marks boundary frames (near tic/no-tic transitions)
as -100 so they are ignored in loss computation.

Inputs:
    configs/paths.yaml   -> new_embeddings_dir
    configs/model.yaml   -> sequence_length, sequence_stride
    outputs/splits/{split_name}/train.csv etc.
    outputs/splits/{split_name}/filter_report.json
    configs/label_config.json

Usage:
    from s05_dataset import TicDataset
    dataset = TicDataset(split_csv, embeddings_dir, filter_report, cfg)
"""

import os
import sys
from pathlib import Path
import json
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset

HOME_DIR = os.environ.get(
    "HOME_DIR",
    "/home/kzaveri1/codes/modular_pipline_package/tic_detection"
)
sys.path.insert(0, HOME_DIR)

with open(os.path.join(HOME_DIR, "configs", "paths.yaml"), "r") as f:
    paths_cfg = yaml.safe_load(f)

with open(os.path.join(HOME_DIR, "configs", "model.yaml"), "r") as f:
    model_cfg = yaml.safe_load(f)


class TicDataset(Dataset):
    def __init__(
        self,
        split_csv: str,
        embeddings_dir: str,
        filter_report_path: str,
        label_config_path: str,
        sequence_length: int = 500,
        sequence_stride: int = 500,
        cache_dir: str = None,
        boundary_frames: int = 0,
    ):
        """
        Args:
            split_csv:           path to train.csv / val.csv / test.csv
            embeddings_dir:      path to Embeddings_new/
            filter_report_path:  path to filter_report.json
            label_config_path:   path to label_config.json
            sequence_length:     number of frames per sequence
            sequence_stride:     step between sequence start points
            cache_dir:           directory to cache built sequences
            boundary_frames:     frames to mark as -100 near tic boundaries
                                 (50 frames = 1 second at 20ms per frame)
        """
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.boundary_frames = boundary_frames

        # -- load configs --
        with open(filter_report_path) as f:
            filter_report = json.load(f)
        with open(label_config_path) as f:
            label_config = json.load(f)

        self.excluded_groups = set(filter_report["excluded_groups"])
        self.type_to_int     = {int(k): v for k, v in label_config["type_to_int"].items()}
        self.no_tic_int      = self.type_to_int[label_config["no_tic_label"]]

        # -- cache path — includes boundary_frames so different settings
        #    don't share cache --
        if cache_dir is not None:
            cache_dir  = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            split_name = Path(split_csv).stem
            cache_path = (
                cache_dir /
                f"{split_name}_seq{sequence_length}_str{sequence_stride}_b{boundary_frames}.pt"
            )
        else:
            cache_path = None

        # -- load from cache or build --
        if cache_path is not None and cache_path.exists():
            print(f"[s05] Loading from cache: {cache_path}")
            self.sequences = torch.load(cache_path, weights_only=False)
            print(f"[s05] {len(self.sequences)} sequences loaded from cache")
        else:
            import pandas as pd
            filenames      = pd.read_csv(split_csv)["filename"].tolist()
            self.sequences = []
            self._cache_and_index(filenames, Path(embeddings_dir))
            print(f"[s05] {len(self.sequences)} sequences built")

            if cache_path is not None:
                torch.save(self.sequences, cache_path)
                print(f"[s05] Cache saved to: {cache_path}")

        tic_seqs = sum(
            1 for _, labels in self.sequences
            if ((labels != self.no_tic_int) & (labels != -100)).any()
        )
        print(f"[s05] Sequences with at least one tic frame: {tic_seqs}")

    def _mark_boundary_frames(
        self,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mark frames within boundary_frames of a tic/no-tic transition
        as -100 so they are ignored in loss computation.

        A frame is a boundary frame if its local window contains both
        tic and no-tic labels.

        Args:
            labels: [N] tensor of frame labels

        Returns:
            labels: [N] with boundary frames set to -100
        """
        labels  = labels.clone()
        n       = len(labels)
        is_tic  = (labels != self.no_tic_int)
        b       = self.boundary_frames

        for i in range(n):
            left  = max(0, i - b)
            right = min(n, i + b + 1)

            window_has_tic    = is_tic[left:right].any().item()
            window_has_no_tic = (~is_tic[left:right]).any().item()

            if window_has_tic and window_has_no_tic:
                labels[i] = -100

        return labels

    def _cache_and_index(self, filenames: list, embeddings_dir: Path) -> None:
        """
        Load all .pt files, filter excluded groups, optionally mark
        boundary frames, build sequences and store in self.sequences.
        """
        for filename in filenames:
            stem    = Path(filename).stem
            patient = stem.split("_")[0]
            pt_path = embeddings_dir / patient / f"{stem}.pt"

            if not pt_path.exists():
                print(f"[s05] ⚠️  PT file not found, skipping: {pt_path}")
                continue

            frames = torch.load(pt_path, map_location="cpu", weights_only=False)
            frames = [f for f in frames if f["Group"] not in self.excluded_groups]

            if len(frames) < self.sequence_length:
                continue

            embeddings = torch.stack([f["embedding"] for f in frames])
            labels     = torch.tensor(
                [self.type_to_int.get(f["Type"], self.no_tic_int) for f in frames],
                dtype=torch.long
            )

            # mark boundary frames as -100
            if self.boundary_frames > 0:
                labels = self._mark_boundary_frames(labels)

            n_frames = embeddings.shape[0]
            for start in range(0, n_frames - self.sequence_length + 1, self.sequence_stride):
                end = start + self.sequence_length
                self.sequences.append((
                    embeddings[start:end],
                    labels[start:end],
                ))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple:
        return self.sequences[idx]


def test_dataset() -> None:
    """
    Quick test to verify TicDataset loads correctly.
    Uses file_split train.csv.
    """
    output_dir     = Path(paths_cfg["output_dir"])
    embeddings_dir = paths_cfg["new_embeddings_dir"]

    split_csv          = output_dir / "splits" / "file_split" / "train.csv"
    filter_report_path = output_dir / "splits" / "file_split" / "filter_report.json"
    label_config_path  = Path(HOME_DIR) / "configs" / "label_config.json"
    boundary_frames    = model_cfg.get("boundary_frames", 0)

    print(f"[test] Building dataset (boundary_frames={boundary_frames})...")
    dataset = TicDataset(
        split_csv          = str(split_csv),
        embeddings_dir     = embeddings_dir,
        filter_report_path = str(filter_report_path),
        label_config_path  = str(label_config_path),
        sequence_length    = model_cfg["sequence_length"],
        sequence_stride    = model_cfg["train_stride"],
        boundary_frames    = boundary_frames,
    )

    print(f"\n[test] Dataset size:      {len(dataset)}")
    emb, labels = dataset[0]
    print(f"[test] Embedding shape:   {emb.shape}")
    print(f"[test] Labels shape:      {labels.shape}")
    print(f"[test] Embedding dtype:   {emb.dtype}")
    print(f"[test] Labels dtype:      {labels.dtype}")
    print(f"[test] Unique labels:     {sorted(labels.unique().tolist())}")
    print(f"[test] Sample labels:     {labels[:10].tolist()}")

    # find first sequence with boundary frames
    if boundary_frames > 0:
        for i in range(len(dataset)):
            _, lbls = dataset[i]
            if (lbls == -100).any():
                print(f"\n[test] First sequence with boundary frames: idx={i}")
                print(f"[test] Boundary frame count: {(lbls == -100).sum().item()}")
                print(f"[test] Tic frame count:      {((lbls != dataset.no_tic_int) & (lbls != -100)).sum().item()}")
                print(f"[test] No-tic frame count:   {(lbls == dataset.no_tic_int).sum().item()}")
                break

    tic_count    = sum(
        1 for i in range(len(dataset))
        if ((dataset[i][1] != dataset.no_tic_int) & (dataset[i][1] != -100)).any()
    )
    no_tic_count = len(dataset) - tic_count
    print(f"\n[test] Sequences with tic:    {tic_count}")
    print(f"[test] Sequences without tic: {no_tic_count}")
    print(f"[test] Tic sequence ratio:    {tic_count/len(dataset)*100:.1f}%")
    print(f"\n[test] Done.")


if __name__ == "__main__":
    test_dataset()