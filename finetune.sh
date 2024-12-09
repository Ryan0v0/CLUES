#!/bin/zsh

current_time=$(date +"%m%d_%H-%M")
lang_list=("Japanese" "Russian" "Spanish")


for lang in "${lang_list[@]}"; do
    dir_path="./output_bad_1.0/${lang}"
    mkdir -p "$dir_path"
    python finetune.py \
        --model_name_or_path  ''\
        --train_files '' \
        --val_files '' \
        --output_dir $dir_path \
        --do_train True \
        --max_seq_length 2048 \
        --logging_steps 1 \
        --save_strategy "steps" \
        --save_steps 65 \
        --max_steps 260 \
        --learning_rate 2e-5 \
        --evaluation_strategy "steps" \
        --eval_steps 20 \
        --lora True \
        --lora_r 16 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules q_proj v_proj \
        --report_to wandb \
        --optim adamw_torch \
        --overwrite_output_dir True \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --warmup_ratio 0.03 \
        --overwrite_output_dir True \
    > ${dir_path}/output_${lang}_1052_${current_time}.txt
done
