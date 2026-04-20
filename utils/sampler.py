"""
utils/sampler.py
----------------
Batched samplers for handling class imbalance in tic detection.

Two strategies:
    BatchedOversampleSampler:
        - Each batch: 50% tic, 50% no-tic
        - Tic sequences sampled with replacement
        - No-tic sequences sampled without replacement

    BatchedUndersampleSampler:
        - Reduces no-tic pool to 3x tic sequences globally
        - Then applies same 50/50 batched sampling on top
        - No-tic pool reshuffled each epoch

Both samplers work at sequence level — a sequence is "tic" if it
contains at least one tic frame.
"""

import random
import torch
from torch.utils.data import Sampler


NO_TIC_INT = 71  # default, overridden at init from label_config
"""
utils/sampler.py
----------------
Batched samplers for handling class imbalance in tic detection.

Two strategies:
    BatchedOversampleSampler:
        - Each batch: 50% tic, 50% no-tic
        - Tic sequences sampled with replacement
        - No-tic sequences sampled without replacement

    BatchedUndersampleSampler:
        - Reduces no-tic pool to 3x tic sequences globally
        - Then applies same 50/50 batched sampling on top
        - No-tic pool reshuffled each epoch
"""

import random
from torch.utils.data import Sampler


class BatchedOversampleSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        no_tic_int: int,
        seed: int = 42,
    ):
        """
        Each batch has exactly 50% tic and 50% no-tic sequences.
        Tic sequences are oversampled with replacement.
        No-tic sequences are sampled without replacement.

        Args:
            dataset:    TicDataset instance
            batch_size: number of sequences per batch
            no_tic_int: integer label for no-tic class
            seed:       random seed
        """
        self.batch_size  = batch_size
        self.seed        = seed
        self.half        = batch_size // 2

        # -- split indices into tic and no-tic --
        self.tic_indices    = []
        self.no_tic_indices = []

        for i in range(len(dataset)):
            _, labels = dataset[i]
            if (labels != no_tic_int).any():
                self.tic_indices.append(i)
            else:
                self.no_tic_indices.append(i)

        print(f"[sampler] Oversample | tic={len(self.tic_indices)} no_tic={len(self.no_tic_indices)}")

        # number of batches determined by no-tic pool size
        self.n_batches = len(self.no_tic_indices) // self.half

    def __iter__(self):
        rng = random.Random(self.seed)

        # shuffle no-tic pool each epoch
        no_tic = self.no_tic_indices.copy()
        rng.shuffle(no_tic)

        indices = []
        for i in range(self.n_batches):
            # sample tic with replacement
            tic_batch = rng.choices(self.tic_indices, k=self.half)
            # take next no-tic without replacement
            no_tic_batch = no_tic[i * self.half:(i + 1) * self.half]
            # combine and shuffle within batch
            batch = tic_batch + no_tic_batch
            rng.shuffle(batch)
            indices.extend(batch)

        return iter(indices)

    def __len__(self):
        return self.n_batches * self.batch_size
class BatchedUndersampleSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        no_tic_int: int,
        undersample_ratio: int = 3,
        seed: int = 42,
    ):
        """
        Reduces no-tic pool to undersample_ratio x tic sequences globally.
        Then applies same 50/50 batched sampling per batch.
        No-tic pool is reshuffled each epoch so different sequences
        are seen over time.

        Args:
            dataset:           TicDataset instance
            batch_size:        number of sequences per batch
            no_tic_int:        integer label for no-tic class
            undersample_ratio: no-tic pool = ratio x tic sequences
            seed:              random seed
        """
        self.batch_size        = batch_size
        self.seed              = seed
        self.half              = batch_size // 2
        self.undersample_ratio = undersample_ratio

        # -- split indices into tic and no-tic --
        self.tic_indices    = []
        self.no_tic_indices = []

        for i in range(len(dataset)):
            _, labels = dataset[i]
            if (labels != no_tic_int).any():
                self.tic_indices.append(i)
            else:
                self.no_tic_indices.append(i)

        # -- reduce no-tic pool to ratio x tic --
        self.no_tic_pool_size = min(
            len(self.no_tic_indices),
            undersample_ratio * len(self.tic_indices)
        )

        print(
            f"[sampler] Undersample | "
            f"tic={len(self.tic_indices)} "
            f"no_tic_full={len(self.no_tic_indices)} "
            f"no_tic_pool={self.no_tic_pool_size} "
            f"(ratio={undersample_ratio}x)"
        )

        # number of batches determined by undersampled no-tic pool
        self.n_batches = self.no_tic_pool_size // self.half

    def __iter__(self):
        rng = random.Random(self.seed)

        # reshuffle full no-tic pool and take first no_tic_pool_size
        no_tic_full = self.no_tic_indices.copy()
        rng.shuffle(no_tic_full)
        no_tic = no_tic_full[:self.no_tic_pool_size]

        indices = []
        for i in range(self.n_batches):
            # oversample tic with replacement
            tic_batch    = rng.choices(self.tic_indices, k=self.half)
            # take next no-tic without replacement from reduced pool
            no_tic_batch = no_tic[i * self.half:(i + 1) * self.half]
            # combine and shuffle within batch
            batch = tic_batch + no_tic_batch
            rng.shuffle(batch)
            indices.extend(batch)

        return iter(indices)

    def __len__(self):
        return self.n_batches * self.batch_size
if __name__ == "__main__":
    import sys
    from pathlib import Path
    import yaml
    sys.path.insert(0, '/home/kzaveri1/codes/modular_pipline_package/tic_detection')
    sys.path.insert(0, '/home/kzaveri1/codes/modular_pipline_package/tic_detection/stages')
    from s05_dataset import TicDataset

    HOME_DIR = '/home/kzaveri1/codes/modular_pipline_package/tic_detection'
    with open(f'{HOME_DIR}/configs/paths.yaml') as f:
        paths_cfg = yaml.safe_load(f)
    with open(f'{HOME_DIR}/configs/model.yaml') as f:
        model_cfg = yaml.safe_load(f)

    output_dir = Path(paths_cfg['output_dir'])
    dataset = TicDataset(
        split_csv          = str(output_dir / 'splits' / 'file_split' / 'train.csv'),
        embeddings_dir     = paths_cfg['new_embeddings_dir'],
        filter_report_path = str(output_dir / 'splits' / 'file_split' / 'filter_report.json'),
        label_config_path  = f'{HOME_DIR}/configs/label_config.json',
        sequence_length    = model_cfg['sequence_length'],
        sequence_stride    = model_cfg['train_stride'],
    )

    print("\n── Oversample Sampler ──")
    over_sampler = BatchedOversampleSampler(
        dataset, batch_size=32, no_tic_int=dataset.no_tic_int
    )
    print(f"Total indices: {len(over_sampler)}")
    print(f"Total batches: {over_sampler.n_batches}")

    print("\n── Undersample Sampler ──")
    under_sampler = BatchedUndersampleSampler(
        dataset, batch_size=32, no_tic_int=dataset.no_tic_int, undersample_ratio=3
    )
    print(f"Total indices: {len(under_sampler)}")
    print(f"Total batches: {under_sampler.n_batches}")