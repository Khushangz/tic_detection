#!/bin/bash
module load anaconda3/2023.09
source activate tic_base

tmux new-session -d -s experiments2

python /home/kzaveri1/codes/modular_pipline_package/tic_detection/cli.py all --exp exp_05 \
  --loss binary \
  --dropout 0.3 \
  --lr 0.001 \
  --sampler batched_undersample \
  --undersample 3 \
  --epochs 20 \
  --class-weights

python /home/kzaveri1/codes/modular_pipline_package/tic_detection/cli.py all --exp exp_06 \
  --loss binary \
  --dropout 0.5 \
  --lr 0.0001 \
  --sampler batched_undersample \
  --undersample 3 \
  --epochs 20 \
  --class-weights 

