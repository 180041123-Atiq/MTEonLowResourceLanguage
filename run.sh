python train.py \
    --model llama2 \
    --prompt ag \
    --epochs 3 \
    --batch 2 \
    --train-path train_comet_da_scaled.csv \
    --test-path test_comet_da_scaled.csv \
    --log-path logs/ag.txt