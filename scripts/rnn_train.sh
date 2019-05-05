#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m  src.bin.train \
    --model_name "dl4mt" \
    --reload \
    --config_path "./configs/dl4mt_config.yaml" \
    --log_path "./scripts/log" \
    --saveto "./scripts/save/" \
    --use_gpu
