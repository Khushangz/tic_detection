"""
stages/s02_extract.py
...
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torch
import torchaudio
import yaml
import logging
from tqdm import tqdm
from transformers import WavLMModel, AutoFeatureExtractor

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

# ── constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
MODEL_NAME       = audio_cfg["model_name"]
FRAME_DURATION_S = audio_cfg["frame_duration_ms"] / 1000
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

def _load_model():
    print(f"[s02] Loading model: {MODEL_NAME}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = WavLMModel.from_pretrained(MODEL_NAME, use_safetensors=True)
    model.eval()
    model.to(DEVICE)
    print(f"[s02] Model loaded on {DEVICE}")
    return model, feature_extractor

def _extract_frames(
    model,
    feature_extractor,
    chunk_path: Path,
    chunk_meta: dict,
) -> list[dict]:
    """
    Run one 30s chunk through WavLM and return one dict per 20ms frame.
    Duplicates the last frame to ensure exactly 1500 frames per chunk,
    covering the full 30s with no gaps between chunks.

    Args:
        model:             WavLMModel in eval mode
        feature_extractor: AutoFeatureExtractor
        chunk_path:        path to the .wav chunk file
        chunk_meta:        dict with ID, Sess, Phase, filename,
                           chunk_idx, start_time_s, end_time_s

    Returns:
        list of dicts, one per 20ms frame:
            ID, Sess, Phase, filename, chunk_idx,
            frame_idx, start_time_s, end_time_s,
            embedding: tensor [D]
    """
    # -- load audio --
    waveform, sr = torchaudio.load(str(chunk_path))
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sr} for {chunk_path}")
    waveform = waveform.squeeze(0).numpy()  # [num_samples]

    # -- preprocess --
    inputs = feature_extractor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # -- forward pass, extract last layer --
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1].squeeze(0)  # [1499, D]

    # -- duplicate last frame to get exactly 1500 frames --
    last_hidden = torch.cat(
        [last_hidden, last_hidden[-1].unsqueeze(0)], dim=0
    )  # [1500, D]

    # -- build one entry per frame --
    frames = []
    chunk_start = chunk_meta["start_time_s"]

    for frame_idx in range(last_hidden.shape[0]):
        frame_start = chunk_start + frame_idx * FRAME_DURATION_S
        frame_end   = frame_start + FRAME_DURATION_S

        frames.append({
            "ID":           chunk_meta["ID"],
            "Sess":         chunk_meta["Sess"],
            "Phase":        chunk_meta["Phase"],
            "filename":     chunk_meta["filename"],
            "chunk_idx":    chunk_meta["chunk_idx"],
            "frame_idx":    frame_idx,
            "start_time_s": round(frame_start, 6),
            "end_time_s":   round(frame_end, 6),
            "embedding":    last_hidden[frame_idx].cpu(),
        })

    return frames


def run_extract() -> None:
    """
    Run the full extraction stage.
    Reads chunks.csv, runs each chunk through WavLM,
    saves one .pt file per audio file containing all 20ms frame embeddings.
    Logs to outputs/logs/s02_extract.log and terminal simultaneously.
    """
    import logging

    # -- setup logging --
    log_dir  = Path(paths_cfg["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "s02_extract.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ]
    )
    log = logging.getLogger(__name__)

    # -- setup dirs --
    embeddings_dir = Path(paths_cfg["new_embeddings_dir"])
    meta_dir       = Path(paths_cfg["output_dir"]) / "meta"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # -- load chunks.csv --
    chunks_path = meta_dir / "chunks.csv"
    log.info(f"[s02] Loading chunks: {chunks_path}")
    chunks_df = pd.read_csv(chunks_path)
    log.info(f"[s02] {len(chunks_df)} chunks to process")

    # -- load model --
    model, feature_extractor = _load_model()

    # -- group by filename so we save one .pt per audio file --
    grouped = chunks_df.groupby("filename")
    log.info(f"[s02] {len(grouped)} unique files to process")

    for filename, group in tqdm(grouped, desc="Extracting"):
        row0      = group.iloc[0]
        save_dir  = embeddings_dir / str(row0["ID"])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{Path(filename).stem}.pt"

        # skip if already done
        if save_path.exists():
            log.info(f"[s02] Skipping (already exists): {save_path}")
            continue

        all_frames = []

        for _, chunk_row in group.iterrows():
            chunk_path = Path(chunk_row["chunk_path"])
            if not chunk_path.exists():
                log.info(f"[s02] ⚠️  Chunk not found, skipping: {chunk_path}")
                continue

            try:
                frames = _extract_frames(
                    model,
                    feature_extractor,
                    chunk_path,
                    chunk_row.to_dict(),
                )
                all_frames.extend(frames)
            except Exception as e:
                log.info(f"[s02] ❌ Error on {chunk_path}: {e}")
                continue

        if len(all_frames) == 0:
            log.info(f"[s02] ⚠️  No frames extracted for {filename}, skipping save")
            continue

        torch.save(all_frames, save_path)
        log.info(f"[s02] ✅ Saved {len(all_frames)} frames → {save_path}")

    log.info(f"[s02] Done.")


if __name__ == "__main__":
    run_extract()

