#!/bin/bash
# Usage:
# ./run_experiment.sh --exp exp_03 --loss multiclass --dropout 0.5 --lr 0.0001 --sampler batched_undersample
# ./run_experiment.sh --exp exp_03 --stage train
# ./run_experiment.sh --exp exp_03 --stage eval

EXP="exp_01"
STAGE="train"
LOSS="multiclass"
DROPOUT="0.3"
LR="0.001"
SAMPLER="batched_oversample"
UNDERSAMPLE_RATIO="3"
HIDDEN_SIZE="256"
EPOCHS="50"

while [[ $# -gt 0 ]]; do
    case $1 in
        --exp)             EXP="$2";              shift 2 ;;
        --stage)           STAGE="$2";            shift 2 ;;
        --loss)            LOSS="$2";             shift 2 ;;
        --dropout)         DROPOUT="$2";          shift 2 ;;
        --lr)              LR="$2";               shift 2 ;;
        --sampler)         SAMPLER="$2";          shift 2 ;;
        --undersample)     UNDERSAMPLE_RATIO="$2"; shift 2 ;;
        --hidden)          HIDDEN_SIZE="$2";      shift 2 ;;
        --epochs)          EPOCHS="$2";           shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "══════════════════════════════════════"
echo " Experiment : $EXP"
echo " Stage      : $STAGE"
echo " Loss       : $LOSS"
echo " Dropout    : $DROPOUT"
echo " LR         : $LR"
echo " Sampler    : $SAMPLER"
echo "══════════════════════════════════════"

# patch model.yaml with experiment params
python3 - <<PYEOF
import yaml, os
HOME_DIR = os.environ.get("HOME_DIR", "/home/kzaveri1/codes/modular_pipline_package/tic_detection")
cfg_path = f"{HOME_DIR}/configs/model.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg["loss"]                = "$LOSS"
cfg["dropout"]             = float("$DROPOUT")
cfg["lr"]                  = float("$LR")
cfg["imbalance_strategy"]  = "$SAMPLER"
cfg["undersample_ratio"]   = int("$UNDERSAMPLE_RATIO")
cfg["bilstm"]["hidden_size"] = int("$HIDDEN_SIZE")
cfg["bilstm"]["dropout"]   = float("$DROPOUT")
cfg["epochs"]              = int("$EPOCHS")
with open(cfg_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
print(f"[run] model.yaml updated")
PYEOF

# run the requested stage
cd /home/kzaveri1/codes/modular_pipline_package/tic_detection/stages

if [ "$STAGE" = "train" ]; then
    python s06_train.py --exp $EXP

elif [ "$STAGE" = "eval" ]; then
    python s07_eval.py --exp $EXP

elif [ "$STAGE" = "all" ]; then
    python s06_train.py --exp $EXP
    python s07_eval.py --exp $EXP

else
    echo "Unknown stage: $STAGE"
    exit 1
fi