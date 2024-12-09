#!/bin/zsh

output_notation='noproj'
model_name_or_path=''
max_token_length=2048
proj_d=8192
seed=42
overwrite=True
checkpoint_path=''
train_set_path=''

tr_lang="English"
scoring_weight=0

python Calculate_grad.py \
    --output_notation=$output_notation \
    --model_name_or_path=$model_name_or_path \
    --max_token_length=$max_token_length \
    --proj_d=$proj_d \
    --seed=$seed \
    --overwrite=$overwrite \
    --checkpoint_path=$checkpoint_path \
    --train_set_path=$train_set_path 


python Compute_score.py \
    --tr_lang=$tr_lang \
    --output_notation=$output_notation \
    --scoring_weight=$scoring_weight \
    --checkpoint_path=$checkpoint_path \
    --train_set_path=$train_set_path \
    --score_save_path=$train_set_path

### Baseline methods: 

# IFD/PPL score
save_name='./IFD_out/temp_ppl.pt'
python inference_loss.py \
    --data_path=$train_set_path \
    --save_name=$save_name \
    --model_name_or_path=$model_name_or_path \
    --checkpoint_path=$checkpoint_path \
    --prompt='alpaca' \
    --mod='cherry' 

python IFD_PPL_score.py \
    --pt_data_path=$save_name  \
    --json_data_path=$train_set_path \
    --json_save_path=$train_set_path \
    --model_name_or_path=$model_name_or_path \
    --checkpoint_path=$checkpoint_path 

# Datainf score
python DataInf.py \
    --tr_lang=$tr_lang \
    --train_set_path=$train_set_path \
    --score_save_path=$train_set_path
