"""
stages/s05_dataset.py
---------------------
Stage 05: Dataset

PyTorch Dataset class that reads labeled .pt files from
Embeddings_new/ and serves sequences to the BiLSTM.

Caches all .pt files in memory on init for fast access.
Skips frames whose Group is in filter_report excluded_groups.
Sequences never cross file boundaries.

Inputs:
    configs/paths.yaml   -> new_embeddings_dir
    configs/model.yaml   -> sequence_length, sequence_stride, normalize
    outputs/splits/{split_name}/train.csv etc.
    outputs/splits/{split_name}/filter_report.json
    outputs/configs/label_config.json

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
        sequence_length: int = 50,
        sequence_stride: int = 25,
        cache_dir: str = None,
    ):
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride

        # -- load configs --
        with open(filter_report_path) as f:
            filter_report = json.load(f)
        with open(label_config_path) as f:
            label_config = json.load(f)

        self.excluded_groups = set(filter_report["excluded_groups"])
        self.type_to_int     = {int(k): v for k, v in label_config["type_to_int"].items()}
        self.no_tic_int      = self.type_to_int[label_config["no_tic_label"]]

        # -- cache path --
        if cache_dir is not None:
            cache_dir  = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            split_name = Path(split_csv).stem
            cache_path = cache_dir / f"{split_name}_seq{sequence_length}_str{sequence_stride}.pt"
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

        tic_seqs = sum(1 for _, labels in self.sequences if (labels != self.no_tic_int).any())
        print(f"[s05] Sequences with at least one tic frame: {tic_seqs}")

    def _cache_and_index(self, filenames: list, embeddings_dir: Path) -> None:
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
    output_dir    = Path(paths_cfg["output_dir"])
    embeddings_dir = paths_cfg["new_embeddings_dir"]

    split_csv          = output_dir / "splits" / "file_split" / "train.csv"
    filter_report_path = output_dir / "splits" / "file_split" / "filter_report.json"
    label_config_path = Path(HOME_DIR) / "configs" / "label_config.json"

    print(f"[test] Building dataset...")
    dataset = TicDataset(
        split_csv          = str(split_csv),
        embeddings_dir     = embeddings_dir,
        filter_report_path = str(filter_report_path),
        label_config_path  = str(label_config_path),
        sequence_length    = model_cfg["sequence_length"],
        sequence_stride    = model_cfg["train_stride"],
    )

    print(f"\n[test] Dataset size:      {len(dataset)}")
    emb, labels = dataset[0]
    print(f"[test] Embedding shape:   {emb.shape}")
    print(f"[test] Labels shape:      {labels.shape}")
    print(f"[test] Embedding dtype:   {emb.dtype}")
    print(f"[test] Labels dtype:      {labels.dtype}")
    print(f"[test] Unique labels:     {labels.unique().tolist()}")
    print(f"[test] Sample labels:     {labels[:10].tolist()}")

    # check a few sequences
    tic_count    = sum(1 for i in range(len(dataset)) if (dataset[i][1] != dataset.no_tic_int).any())
    no_tic_count = len(dataset) - tic_count
    print(f"\n[test] Sequences with tic:    {tic_count}")
    print(f"[test] Sequences without tic: {no_tic_count}")
    print(f"[test] Tic sequence ratio:    {tic_count/len(dataset)*100:.1f}%")
    print(f"\n[test] Done.")


if __name__ == "__main__":
    test_dataset()