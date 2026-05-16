#!/bin/bash
module load anaconda3/2023.09
source activate tic_base

CLI=/home/kzaveri1/codes/modular_pipline_package/tic_detection/cli.py

tmux new-session -d -s experiments3

tmux send-keys -t experiments3 "

python $CLI all --exp bilstm_1l_lr1e4_do05 \
  --model bilstm --num-layers 1 \
  --loss multiclass \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp bilstm_1l_lr1e3_do03 \
  --model bilstm --num-layers 1 \
  --loss multiclass \
  --dropout 0.3 --lr 0.001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp bilstm_2l_lr1e4_do05 \
  --model bilstm --num-layers 2 \
  --loss multiclass \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp bilstm_2l_lr1e3_do03 \
  --model bilstm --num-layers 2 \
  --loss multiclass \
  --dropout 0.3 --lr 0.001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp cnn_1l_lr1e4_do05 \
  --model cnn_bilstm --num-layers 1 \
  --loss multiclass \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp cnn_1l_lr1e3_do03 \
  --model cnn_bilstm --num-layers 1 \
  --loss multiclass \
  --dropout 0.3 --lr 0.001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp cnn_2l_lr1e4_do05 \
  --model cnn_bilstm --num-layers 2 \
  --loss multiclass \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp cnn_2l_lr1e3_do03 \
  --model cnn_bilstm --num-layers 2 \
  --loss multiclass \
  --dropout 0.3 --lr 0.001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30

" Enter

echo "All 8 experiments queued in tmux session 'experiments'"
echo "Attach with: tmux attach -t experiments3"