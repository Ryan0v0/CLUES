#!/bin/zsh

python merging_eval_med.py \
    --model_name_or_path '' \
    --output_dir_for_checkpoints '' \
    --checkpoint_num 260 \
    --data_path '' \
    --max_token_length 2048 \
    --eval_gen \
    --num_samples 2982