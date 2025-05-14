#!/bin/bash

for model in llama2
do
  for i in 1 2 3 4 5
  do
    for prompt in ag dg dag
    do
      echo "Run $i for $model with  prompt: $prompt"
      python engine.py \
        --model "$model" \
        --prompt "$prompt" \
        --epochs 3 \
        --batch 2 \
        --lr 2e-5 \
        --train-path train.csv \
        --val-path val.csv \
        --test-path test.csv \
        --output-path output \
        --log-path "logs/${model}_${prompt}_fine4_run${i}.txt" --cusTok --quantized
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
