#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m  src.bin.train \
    --model_name "transformer" \
    --reload \
    --config_path "./configs/transformer_wmt14_en2de.yaml" \
    --log_path "./scripts/log_tf_en2de" \
    --saveto "./scripts/save_tf_en2de" \
    --use_gpu
