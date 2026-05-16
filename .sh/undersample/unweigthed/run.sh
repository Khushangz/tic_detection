#!/bin/bash
module load anaconda3/2023.09
source activate tic_base

CLI=/home/kzaveri1/codes/modular_pipline_package/tic_detection/cli.py

tmux new-session -d -s experiments_binary

tmux send-keys -t experiments_binary "

python $CLI all --exp bilstm_1l_lr1e4_do05_bin \
  --model bilstm --num-layers 1 \
  --loss binary \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp bilstm_1l_lr1e3_do03_bin \
  --model bilstm --num-layers 1 \
  --loss binary \
  --dropout 0.3 --lr 0.001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp bilstm_2l_lr1e4_do05_bin \
  --model bilstm --num-layers 2 \
  --loss binary \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp bilstm_2l_lr1e3_do03_bin \
  --model bilstm --num-layers 2 \
  --loss binary \
  --dropout 0.3 --lr 0.001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp cnn_1l_lr1e4_do05_bin \
  --model cnn_bilstm --num-layers 1 \
  --loss binary \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp cnn_1l_lr1e3_do03_bin \
  --model cnn_bilstm --num-layers 1 \
  --loss binary \
  --dropout 0.3 --lr 0.001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp cnn_2l_lr1e4_do05_bin \
  --model cnn_bilstm --num-layers 2 \
  --loss binary \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30 && \

python $CLI all --exp cnn_2l_lr1e3_do03_bin \
  --model cnn_bilstm --num-layers 2 \
  --loss binary \
  --dropout 0.3 --lr 0.001 \
  --sampler batched_undersample --undersample 3 \
  --epochs 30

" Enter

echo "All 8 binary experiments queued in tmux session 'experiments_binary'"
echo "Attach with: tmux attach -t experiments_binary"