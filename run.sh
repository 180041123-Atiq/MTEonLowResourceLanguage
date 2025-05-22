#!/bin/bash

for model in llama2 llama213b openchat gemma
do
  for i in 1 2 3 4 5
  do
    for prompt in ag dg dag
    do
      echo "Run $model for $i with prompt: $prompt"
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
