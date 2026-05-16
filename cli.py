"""
cli.py
------
Command-line entry point for the tic_detection pipeline.

Usage:
    python cli.py train --exp exp_03 --loss multiclass --dropout 0.5 --lr 0.0001
    python cli.py eval  --exp exp_03
    python cli.py all   --exp exp_03 --loss multiclass --dropout 0.5

Stages:
    train   run s06_train.py
    eval    run s07_eval.py
    all     run train then eval

Model config flags (patch model.yaml before running):
    --model             bilstm | cnn_bilstm         (default: bilstm)
    --loss              multiclass | binary         (default: multiclass)
    --dropout           float                       (default: 0.3)
    --lr                float                       (default: 0.001)
    --num-layers        int                         (default: 2)
    --hidden            int                         (default: 256)
    --sampler           batched_oversample |        (default: batched_oversample)
                        batched_undersample | none
    --undersample       int                         (default: 3)
    --epochs            int                         (default: 50)
    --batch-size        int                         (default: 32)
    --scheduler         cosine | step | plateau     (default: cosine)
    --class-weights     flag, no value needed       (default: false)
    --seq-len           int                         (default: 500)
"""

import os
import sys
import argparse
import yaml
import subprocess
from pathlib import Path

HOME_DIR = os.environ.get(
    "HOME_DIR",
    "/home/kzaveri1/codes/modular_pipline_package/tic_detection"
)

MODEL_YAML = os.path.join(HOME_DIR, "configs", "model.yaml")
STAGES_DIR = os.path.join(HOME_DIR, "stages")


# ---------------------------------------------------------------------------
# Patch model.yaml with CLI overrides
# ---------------------------------------------------------------------------

def patch_model_yaml(args) -> None:
    with open(MODEL_YAML) as f:
        cfg = yaml.safe_load(f)

    if args.model       is not None: cfg["model_type"]           = args.model
    if args.loss        is not None: cfg["loss"]                 = args.loss
    if args.lr          is not None: cfg["lr"]                   = float(args.lr)
    if args.epochs      is not None: cfg["epochs"]               = int(args.epochs)
    if args.batch_size  is not None: cfg["batch_size"]           = int(args.batch_size)
    if args.sampler     is not None: cfg["imbalance_strategy"]   = args.sampler
    if args.undersample is not None: cfg["undersample_ratio"]    = int(args.undersample)
    if args.scheduler   is not None: cfg["scheduler"]            = args.scheduler
    if args.seq_len     is not None: cfg["sequence_length"]      = int(args.seq_len)
    if args.class_weights:           cfg["use_class_weights"]    = True
    if args.hidden      is not None: cfg["bilstm"]["hidden_size"] = int(args.hidden)
    if args.num_layers  is not None: cfg["bilstm"]["num_layers"]  = int(args.num_layers)

    if args.dropout is not None:
        cfg["bilstm"]["dropout"] = float(args.dropout)
        cfg["dropout"]           = float(args.dropout)

    with open(MODEL_YAML, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"[cli] model.yaml patched:")
    print(f"      model={cfg.get('model_type')}  "
          f"loss={cfg.get('loss')}  "
          f"lr={cfg.get('lr')}  "
          f"dropout={cfg['bilstm'].get('dropout')}  "
          f"num_layers={cfg['bilstm'].get('num_layers')}  "
          f"epochs={cfg.get('epochs')}  "
          f"sampler={cfg.get('imbalance_strategy')}  "
          f"batch_size={cfg.get('batch_size')}")


# ---------------------------------------------------------------------------
# Print experiment summary
# ---------------------------------------------------------------------------

def print_summary(args) -> None:
    print("\n" + "═" * 50)
    print(f"  Experiment : {args.exp}")
    print(f"  Stage      : {args.stage}")
    if args.model:       print(f"  Model      : {args.model}")
    if args.loss:        print(f"  Loss       : {args.loss}")
    if args.lr:          print(f"  LR         : {args.lr}")
    if args.dropout:     print(f"  Dropout    : {args.dropout}")
    if args.num_layers:  print(f"  Num layers : {args.num_layers}")
    if args.sampler:     print(f"  Sampler    : {args.sampler}")
    if args.epochs:      print(f"  Epochs     : {args.epochs}")
    if args.batch_size:  print(f"  Batch size : {args.batch_size}")
    print("═" * 50 + "\n")


# ---------------------------------------------------------------------------
# Run a stage
# ---------------------------------------------------------------------------

def run_stage(script: str, exp: str, eval_suffix: str = "") -> None:
    cmd = [sys.executable, os.path.join(STAGES_DIR, script), "--exp", exp]
    if eval_suffix and script == "s07_eval.py":
        cmd += ["--eval-suffix", eval_suffix]
    print(f"[cli] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=STAGES_DIR)
    if result.returncode != 0:
        print(f"[cli] ❌ {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="tic_detection pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("stage", choices=["train", "eval", "all"],
                        help="which stage to run")
    parser.add_argument("--exp",          type=str,   default="exp_01",
                        help="experiment name (default: exp_01)")

    # model config overrides
    parser.add_argument("--model",        type=str,   default=None,
                        help="bilstm | cnn_bilstm")
    parser.add_argument("--loss",         type=str,   default=None,
                        help="loss type: multiclass | binary")
    parser.add_argument("--dropout",      type=float, default=None,
                        help="dropout rate")
    parser.add_argument("--lr",           type=float, default=None,
                        help="learning rate")
    parser.add_argument("--num-layers",   type=int,   default=None,
                        dest="num_layers", help="number of BiLSTM layers")
    parser.add_argument("--hidden",       type=int,   default=None,
                        help="BiLSTM hidden size")
    parser.add_argument("--epochs",       type=int,   default=None,
                        help="number of training epochs")
    parser.add_argument("--batch-size",   type=int,   default=None,
                        dest="batch_size", help="batch size")
    parser.add_argument("--sampler",      type=str,   default=None,
                        help="batched_oversample | batched_undersample | none")
    parser.add_argument("--undersample",  type=int,   default=None,
                        help="undersample ratio (default: 3)")
    parser.add_argument("--scheduler",    type=str,   default=None,
                        help="cosine | step | plateau | none")
    parser.add_argument("--class-weights", action="store_true",
                        dest="class_weights", default=False,
                        help="enable class weights in loss")
    parser.add_argument("--seq-len",      type=int,   default=None,
                        dest="seq_len", help="sequence length")
    parser.add_argument("--eval-suffix", type=str, default="",
                    dest="eval_suffix", help="suffix for eval output dir e.g. _no_vote")

    args = parser.parse_args()

    print_summary(args)

    # patch model.yaml if any overrides provided
    has_overrides = any([
        args.model, args.loss, args.dropout, args.lr,
        args.num_layers, args.hidden, args.sampler,
        args.undersample, args.epochs, args.batch_size,
        args.scheduler, args.class_weights, args.seq_len,
    ])
    if has_overrides:
        patch_model_yaml(args)

    # run stage
    if args.stage == "train":
        run_stage("s06_train.py", args.exp)

    elif args.stage == "eval":
        run_stage("s07_eval.py", args.exp, eval_suffix=args.eval_suffix)

    elif args.stage == "all":
        run_stage("s06_train.py", args.exp)
        run_stage("s07_eval.py",  args.exp, eval_suffix=args.eval_suffix)




if __name__ == "__main__":
    main()