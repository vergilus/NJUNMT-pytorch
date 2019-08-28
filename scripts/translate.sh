#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

N=$1

#export THEANO_FLAGS=device=cuda3,floatX=float32,mode=FAST_RUN
export MODEL_NAME="transformer"

python3 ../translate.py \
    --model_name $MODEL_NAME \
    --source_path "/home/public_data/nmtdata/wmt14_en-de_data_selection/newstest20$1.tok.en" \
    --model_path "./save_tf_en2de/$MODEL_NAME.best.final" \
    --config_path "../configs/transformer_wmt14_en2de.yaml" \
    --batch_size 40 \
    --beam_size 5 \
    --saveto "./results/$MODEL_NAME.20$1.out" \
	--keep_n 1 \
    --use_gpu

#detokenize
perl ~/../user_data/zouw/tokenizer/detokenizer.perl -thread 5 -l de < ./results/$MODEL_NAME.20$1.out.0 \
			 > ./results/$MODEL_NAME.20$1.out
