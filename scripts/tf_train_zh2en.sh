#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m  src.bin.train \
    --model_name "transformer" \
    --reload \
    --config_path "./configs/transformer_nist_zh2en_bpe.yaml" \
    --log_path "./scripts/log_tf_zh2en" \
    --saveto "./scripts/save_tf_zh2en" \
    --use_gpu
