"""
stages/s01_chunk.py
-------------------
Stage 01: Chunk

Reads manifest.csv and cuts each raw audio file into 30s chunks.
Saves chunks as .wav files and outputs chunks.csv.

Inputs:
    configs/paths.yaml   -> raw_audio_dir, output_dir
    configs/audio.yaml   -> chunk_duration_s, stride_s, padding_strategy
    outputs/meta/manifest.csv

Outputs:
    outputs/chunks/{ID}/{filename}_chunk{i}.wav
    outputs/meta/chunks.csv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import torchaudio
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

# ── constants from config ────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
CHUNK_DURATION_S = audio_cfg["chunk_duration_s"]
STRIDE_S         = audio_cfg["stride_s"]
PADDING_STRATEGY = audio_cfg["padding_strategy"]
CHUNK_SAMPLES    = CHUNK_DURATION_S * SAMPLE_RATE
STRIDE_SAMPLES   = STRIDE_S * SAMPLE_RATE

def _load_audio(audio_path: Path) -> torch.Tensor:
    """
    Load a .wav file and return waveform as a 1D tensor.
    Expects 16kHz mono. Raises if sample rate does not match.

    Returns:
        waveform: tensor of shape [num_samples]
    """
    waveform, sr = torchaudio.load(str(audio_path))

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Expected sample rate {SAMPLE_RATE}, got {sr} for {audio_path}"
        )

    # convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.squeeze(0)  # [num_samples]

def _chunk_waveform(waveform: torch.Tensor, filename: str) -> list[dict]:
    """
    Cut a waveform into 30s chunks.
    Pads the last chunk if shorter than chunk_duration_s.

    Args:
        waveform: 1D tensor [num_samples]
        filename: base filename for naming chunks

    Returns:
        list of dicts, one per chunk:
            chunk_idx:    int
            start_time_s: float
            end_time_s:   float
            duration_s:   float
            waveform:     tensor [CHUNK_SAMPLES]
    """
    chunks = []
    total_samples = waveform.shape[0]
    chunk_idx = 0
    start = 0

    while start < total_samples:
        end = start + CHUNK_SAMPLES
        chunk_waveform = waveform[start:end]

        # pad last chunk if shorter than CHUNK_SAMPLES
        if chunk_waveform.shape[0] < CHUNK_SAMPLES:
            pad_size = CHUNK_SAMPLES - chunk_waveform.shape[0]
            if PADDING_STRATEGY == "zero":
                chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, pad_size))
            elif PADDING_STRATEGY == "reflect":
                chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, pad_size), mode="reflect")
            elif PADDING_STRATEGY == "edge":
                chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, pad_size), mode="replicate")
            else:
                raise ValueError(f"Unknown padding strategy: {PADDING_STRATEGY}")

        chunks.append({
            "chunk_idx":    chunk_idx,
            "start_time_s": start / SAMPLE_RATE,
            "end_time_s":   min(end, total_samples) / SAMPLE_RATE,
            "duration_s":   chunk_waveform.shape[0] / SAMPLE_RATE,
            "waveform":     chunk_waveform,
        })

        chunk_idx += 1
        start += STRIDE_SAMPLES

    return chunks

def _save_chunks(
    chunks: list[dict],
    row: pd.Series,
    chunks_dir: Path,
) -> list[dict]:
    """
    Save each chunk as a .wav file and return metadata rows for chunks.csv.

    Args:
        chunks:     list of chunk dicts from _chunk_waveform
        row:        manifest row with ID, Sess, Phase, filename
        chunks_dir: outputs/chunks/

    Returns:
        list of metadata dicts, one per chunk
    """
    save_dir = chunks_dir / str(row["ID"])
    save_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(row["filename"]).stem  # e.g. DET0001_V1_LO
    metadata_rows = []

    for chunk in chunks:
        chunk_filename = f"{base_name}_chunk{chunk['chunk_idx']}.wav"
        chunk_path = save_dir / chunk_filename

        torchaudio.save(
            str(chunk_path),
            chunk["waveform"].unsqueeze(0),  # [1, num_samples]
            SAMPLE_RATE,
        )

        metadata_rows.append({
            "ID":           row["ID"],
            "Sess":         row["Sess"],
            "Phase":        row["Phase"],
            "filename":     row["filename"],
            "chunk_idx":    chunk["chunk_idx"],
            "chunk_path":   str(chunk_path),
            "start_time_s": chunk["start_time_s"],
            "end_time_s":   chunk["end_time_s"],
            "duration_s":   chunk["duration_s"],
        })

    return metadata_rows

def run_chunk() -> pd.DataFrame:
    """
    Run the full chunking stage.
    Reads manifest.csv, chunks each audio file, saves .wav chunks
    and outputs chunks.csv.

    Returns:
        chunks_df — saved to outputs/meta/chunks.csv
    """
    # -- setup dirs --
    output_dir = Path(paths_cfg["output_dir"])
    chunks_dir = output_dir / "chunks"
    meta_dir   = output_dir / "meta"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # -- load manifest --
    manifest_path = meta_dir / "manifest.csv"
    print(f"[s01] Loading manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    print(f"[s01] {len(manifest)} files to chunk")

    # -- process each file --
    raw_audio_dir = Path(paths_cfg["raw_audio_dir"])
    all_metadata  = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Chunking"):
        
        primary  = raw_audio_dir / str(row["ID"]) / row["filename"]
        fallback = raw_audio_dir / str(row["ID"]) / row["filename"].replace(".wav", "_Front.wav")

        if primary.exists():
            audio_path = primary
        elif fallback.exists():
            audio_path = fallback
        else:
            print(f"[s01] ⚠️  Audio file not found, skipping: {primary}")
            continue

        try:
            waveform = _load_audio(audio_path)
            chunks   = _chunk_waveform(waveform, row["filename"])
            metadata = _save_chunks(chunks, row, chunks_dir)
            all_metadata.extend(metadata)
        except Exception as e:
            print(f"[s01] ❌ Error processing {row['filename']}: {e}")
            continue

    # -- save chunks.csv --
    chunks_df = pd.DataFrame(all_metadata)
    chunks_csv_path = meta_dir / "chunks.csv"
    chunks_df.to_csv(chunks_csv_path, index=False)

    print(f"[s01] Total chunks created: {len(chunks_df)}")
    print(f"[s01] chunks.csv saved to:  {chunks_csv_path}")
    print(f"[s01] Done.")

    return chunks_df



if __name__ == "__main__":
    run_chunk()