#!/bin/bash

for prompt in dag
do
  for i in 1
  do
    echo "Run $i for prompt: $prompt"
    python train.py \
      --model llama2 \
      --prompt "$prompt" \
      --epochs 3 \
      --batch 2 \
      --train-path train_comet_da_scaled.csv \
      --test-path test_comet_da_scaled.csv \
      --log-path "logs/llama2CusTokRegHead${prompt}_run${i}.txt"
  done
done

#python train.py \
#  --model llama2 \
#  --prompt dag \
#  --epochs 3 \
#  --batch 2 \
#  --train-path train_comet_da_scaled.csv \
#  --test-path test_comet_da_scaled.csv \
#  --log-path "logs/llama2opt_dag.txt"
