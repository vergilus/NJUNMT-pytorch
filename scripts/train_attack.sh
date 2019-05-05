# !/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m  adversarials.main_attack \
	    --n 1 \
	    --config_path "/home/zouw/pycharm_project_NMT_torch/configs/nist_zh2en_attack.yaml" \
		--save_to "./adversarials/attack_log" \
		--use_gpu \
		--share_optim \
		
