# tic_detection

A modular, reproducible pipeline for vocal tic detection from audio using WavLM embeddings and a BiLSTM classifier.

---

## Overview

This package processes raw audio recordings of patients with Tourette syndrome, extracts frame-level WavLM embeddings, labels each 20ms frame against annotated tic intervals, and trains a BiLSTM to detect and classify vocal tics.

The pipeline is designed for reproducibility — every stage is a standalone script driven by config files. Nothing changes between experiments except the configs.

---

## Pipeline Stages

```
s00_inventory      validate inputs, produce manifest
s01_chunk          cut audio into 30s chunks for WavLM
s02_extract        WavLM forward pass → 20ms frame embeddings
s03_label          align tic annotations to frame indices
s03b_group_counts  count tic group frames per file
s04_split          JSD-optimized train/val/test split
s04b_filter        determine which groups to exclude
s05_dataset        build sequence dataset with caching
s06_train          BiLSTM training loop
s07_eval           inference, voting, metrics, plots
```

---

## Directory Structure

```
tic_detection/
├── configs/
│   ├── paths.yaml          cluster-specific paths (not committed)
│   ├── paths_template.yaml template for paths.yaml
│   ├── audio.yaml          chunking and extraction params
│   ├── split.yaml          JSD split parameters
│   ├── model.yaml          BiLSTM architecture and training params
│   ├── eval.yaml           evaluation and voting params
│   ├── tic_groups.csv      TicType → group mapping
│   └── label_config.json   generated label schema (auto-generated)
├── stages/
│   ├── s00_inventory.py
│   ├── s01_chunk.py
│   ├── s02_extract.py
│   ├── s03_label.py
│   ├── s03b_group_counts.py
│   ├── s04_split.py
│   ├── s04b_filter.py
│   ├── s05_dataset.py
│   ├── s06_train.py
│   └── s07_eval.py
├── models/
│   ├── base.py             abstract base classifier
│   ├── bilstm.py           BiLSTM classifier
│   └── factory.py          model factory
├── utils/
│   ├── io.py               file I/O utilities
│   ├── labels.py           label processing utilities
│   ├── sampler.py          batched oversample/undersample samplers
│   ├── metrics.py          evaluation metrics
│   └── audio.py            audio utilities
└── outputs/                generated outputs (not committed)
    ├── meta/               manifest, frames summary, group counts
    ├── splits/             train/val/test split CSVs + dashboard
    ├── datasets/           cached sequence tensors
    └── runs/               training experiments
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Khushangz/tic_detection.git
cd tic_detection
```

**2. Install dependencies**
```bash
pip install torch torchaudio transformers pandas numpy scikit-learn matplotlib pyyaml tqdm scipy
```

**3. Configure paths**
```bash
cp configs/paths_template.yaml configs/paths.yaml
# edit configs/paths.yaml with your cluster paths
```

---

## Config Files

### `configs/paths.yaml`
```yaml
tic_csv: /path/to/filtered_ticList_metadata.csv
embeddings_dir: /path/to/Embeddings          # original embeddings
new_embeddings_dir: /path/to/Embeddings_new  # new per-frame embeddings
raw_audio_dir: /path/to/DET_Audio_Data_16kHz_mono
output_dir: /path/to/tic_detection/outputs
cache_dir: /path/to/sequence_cache
```

### `configs/audio.yaml`
```yaml
chunk_duration_s: 30      # WavLM input window
stride_s: 30              # no overlap
frame_duration_ms: 20     # WavLM output frame size
padding_strategy: zero    # zero | reflect | edge
min_duration_s: 30.0
model_name: microsoft/wavlm-base
```

### `configs/model.yaml`
```yaml
sequence_length: 500
train_stride: 250
eval_stride: 500
normalize: false
model_type: bilstm
bilstm:
  hidden_size: 256
  num_layers: 2
  dropout: 0.3
epochs: 50
batch_size: 32
lr: 0.001
scheduler: cosine
imbalance_strategy: batched_oversample
undersample_ratio: 3
use_class_weights: false
```

### `configs/eval.yaml`
```yaml
split: test
voting:
  enabled: true
  window: 100
  strategy: majority
metrics: [auroc, f1, precision, recall]
per_group: true
save_predictions: true
test_split: file_split
```

---

## Running the Pipeline

Run each stage from the `stages/` directory:

```bash
cd stages/

python s00_inventory.py
python s01_chunk.py
python s02_extract.py       # long — run in tmux
python s03_label.py
python s03b_group_counts.py
python s04_split.py
python s04b_filter.py
python s05_dataset.py
python s06_train.py --exp exp_01
python s07_eval.py  --exp exp_01
```

Generate the split dashboard after `s04_split.py`:
```bash
python generate_dashboard.py
# opens outputs/splits/dashboard.html in browser
```

---

## Data Format

**Input CSV** (`filtered_ticList_metadata.csv`):
```
ID, Sess, Phase, Type, StartTime, EndTime, Duration, Audio_duration
DET0102, 3, LO, 1010, 389.627, 390.567, 0.94, 1087.55
```

**Embedding format** (per-frame `.pt` files after s02+s03):
```python
[
  {
    "ID": "DET0102", "Sess": 3, "Phase": "LO",
    "filename": "DET0102_V3_LO.wav",
    "chunk_idx": 13, "frame_idx": 0,
    "start_time_s": 390.0, "end_time_s": 390.02,
    "embedding": tensor([768]),
    "Type": 1010, "Group": "Throat Clearing"
  },
  ...
]
```

---

## Model

**Architecture:** Bidirectional LSTM with frame-level classification
- Input: `[batch, 500, 768]` — 500 frames × 768-dim WavLM embeddings
- Output: `[batch, 500, 72]` — 72 classes (71 tic types + no-tic)
- Parameters: ~3.7M

**Training:** Two metrics reported per epoch
- Binary AUROC/F1 — tic vs no-tic detection
- Multiclass AUROC/F1 — tic type classification

**Imbalance handling:**
- `batched_oversample` — 50/50 tic/no-tic per batch, tic oversampled with replacement
- `batched_undersample` — no-tic pool reduced to 3× tic count, then 50/50 per batch

---

## Outputs

After training:
```
outputs/runs/exp_01/
    best.pt              best model checkpoint
    config.json          full config snapshot
    metrics.json         per-epoch metrics history
    train.log            training log
    plots/
        loss.png
        binary_auroc.png
        binary_f1.png
        mc_auroc.png
        mc_f1.png
    eval/
        results.json
        per_group_metrics.csv
        confusion_matrix.png
        predictions.csv
```

