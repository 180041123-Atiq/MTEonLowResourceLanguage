#!/bin/bash

for model in llama2
do
  for prompt in refless
  do
    for i in 1
    do
      echo "Run $i for prompt: $prompt"
      python engine.py \
        --model "$model" \
        --prompt "$prompt" \
        --epochs 1 \
        --batch 2 \
        --lr 2e-5 \
        --train-path train.csv \
        --val-path val.csv \
        --test-path test.csv \
        --output-path output \
        --log-path "logs/allinone${prompt}_run${i}.txt" --quantized --cusTok
    done
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
