#!/bin/bash
module load anaconda3/2023.09
source activate tic_base
python /home/kzaveri1/codes/modular_pipline_package/tic_detection/cli.py all --exp exp_03 \
  --loss binary \
  --dropout 0.3 \
  --lr 0.001 \
  --sampler batched_undersample \
  --undersample 3 \
  --epochs 20

python /home/kzaveri1/codes/modular_pipline_package/tic_detection/cli.py all --exp exp_04 \
  --loss binary \
  --dropout 0.5 \
  --lr 0.0001 \
  --sampler batched_undersample \
  --undersample 3 \
  --epochs 20

