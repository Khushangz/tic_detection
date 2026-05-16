#!/bin/bash
module load anaconda3/2023.09
source activate tic_base
 
CLI=/home/kzaveri1/codes/modular_pipline_package/tic_detection/cli.py
 
tmux new-session -d -s seqlen_experiments
 
tmux send-keys -t seqlen_experiments "
 
python $CLI all --exp cnn_2l_bin_seq20 \
  --model cnn_bilstm --num-layers 2 \
  --loss binary \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --seq-len 20 --batch-size 32 \
  --epochs 30 && \
 
python $CLI all --exp cnn_2l_bin_seq50 \
  --model cnn_bilstm --num-layers 2 \
  --loss binary \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --seq-len 50 --batch-size 32 \
  --epochs 30 && \
 
python $CLI all --exp cnn_2l_bin_seq100 \
  --model cnn_bilstm --num-layers 2 \
  --loss binary \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --seq-len 100 --batch-size 32 \
  --epochs 30 && \
 
python $CLI all --exp cnn_2l_bin_seq250 \
  --model cnn_bilstm --num-layers 2 \
  --loss binary \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --seq-len 250 --batch-size 32 \
  --epochs 30 && \
 
python $CLI all --exp cnn_2l_bin_seq500 \
  --model cnn_bilstm --num-layers 2 \
  --loss binary \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --seq-len 500 --batch-size 32 \
  --epochs 30 && \
 
python $CLI all --exp cnn_2l_bin_seq750 \
  --model cnn_bilstm --num-layers 2 \
  --loss binary \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --seq-len 750 --batch-size 16 \
  --epochs 30 && \
 
python $CLI all --exp cnn_2l_bin_seq1000 \
  --model cnn_bilstm --num-layers 2 \
  --loss binary \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --seq-len 1000 --batch-size 16 \
  --epochs 30 && \
 
python $CLI all --exp cnn_2l_bin_seq1500 \
  --model cnn_bilstm --num-layers 2 \
  --loss binary \
  --dropout 0.5 --lr 0.0001 \
  --sampler batched_undersample --undersample 3 \
  --seq-len 1500 --batch-size 8 \
  --epochs 30
 
" Enter
 
echo "Sequence length experiments queued in tmux session 'seqlen_experiments'"
echo "Attach with: tmux attach -t seqlen_experiments"