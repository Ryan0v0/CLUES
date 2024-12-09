# %%
import os
import transformers
from transformers import AutoTokenizer, LlamaTokenizer, DataCollatorForSeq2Seq
from peft import PeftModel, LoraConfig, get_peft_model

import sys
from datasets import load_dataset
from time import time
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle
import torch
import argparse
from trl import DataCollatorForCompletionOnlyLM
from torch.utils.data import DataLoader
import pdb
import json
import numpy as np
from utils import *
from fed import *
from datasets import Dataset
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
# %%
# tr_lang='Chinese'
output_notation='noproj_adam'
lang_list=["Chinese", "English", "French", "Japanese", "Russian", "Spanish"]
checkpoint_nums=['65','131','197','260']
model_name_or_path= '' 
max_token_length=2048
threshold_score=None
select_ratio=0.6
proj_d=8192
seed=42
scoring_weight=1.0
# %%
for tr_lang in lang_list:
    checkpoint_path_list=[f'{tr_lang}/checkpoint-{num}' for num in checkpoint_nums]
    train_set_path= f'data/{tr_lang}_anc.json' 
    eval_set_path_list= [f'data/{lang}_val.json' for lang in lang_list]
    tr_grad_file_path_list=[f'{checkpoint_path}/{output_notation}/anchor_gradients.pkl' for checkpoint_path in checkpoint_path_list]
    val_grad_file_path_checkpoint_list=[[f'{checkpoint_path}/{output_notation}/val_gradients_{lang}.pkl'  for lang in lang_list] for checkpoint_path in checkpoint_path_list]


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with open(train_set_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    train_dataset = Dataset.from_list(train_data)

    val_dataset_list=[]
    for eval_set_path in eval_set_path_list:
        with open(eval_set_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        val_dataset = Dataset.from_list(val_data)
        val_dataset_list.append(val_dataset)

    sorted_tuple = []

    ckpt_scores_list=[]
    ckpt_scores_val_list=[]
    for checkpoint_path,tr_grad_file_path, val_grad_file_path_list in zip(checkpoint_path_list, tr_grad_file_path_list,val_grad_file_path_checkpoint_list): #对不同checkpoint循环
        print('computing', checkpoint_path, flush=True)
        try:
            with open(tr_grad_file_path, 'rb') as file:
                tr_grad_dict = pickle.load(file) 
            val_grad_dict_list=[]
            for val_grad_file_path in val_grad_file_path_list:
                with open(val_grad_file_path, 'rb') as file:
                    val_grad_dict = pickle.load(file) 
                    val_grad_dict_list.append(val_grad_dict)  
        except:
            print("lack grad files", flush=True)

        # Calculate Learning Rate   
        with open(f'{checkpoint_path}/trainer_state.json', 'r', encoding='utf-8') as f:
            LRdict = json.load(f)['log_history']
        LR=torch.tensor([ step['learning_rate'] for step in LRdict if step['epoch']>LRdict[-1]['epoch']-1 ]).mean()
        print("LR",LR , flush=True)

        # Calculate average validation gradient for this checkpoint
        ave_val_grad={}
        for lang, val_grad_dict in zip(lang_list,val_grad_dict_list):
            ave_val_grad[lang]=0
            val_grad_list=list(val_grad_dict.values())
            for val_grad in val_grad_list:
                ave_val_grad[lang]+=val_grad/len(val_grad_list)

        # Calculate scores for each data point (training/validation) in this checkpoint
        ckpt_scores = []    
        for id in tqdm(range(len(tr_grad_dict))):
            score=0
            for lang in lang_list:
                if lang==tr_lang:
                    score+=LR * (tr_grad_dict[id] * ave_val_grad[lang]).sum()
                else:
                    score+=LR * (tr_grad_dict[id] * ave_val_grad[lang]).sum() * scoring_weight
            ckpt_scores.append(score)

        
        ckpt_scores_list.append(torch.tensor(ckpt_scores))

        # print(tr_grad_dict[id])

    scores=torch.stack(ckpt_scores_list).mean(dim=0)

    # pdb.set_trace()


    # Convert scores list to numpy array
    scores_array = np.array(scores)

    # Add 'score' column using add_column() method
    train_dataset= train_dataset.add_column('score', scores_array)

    train_data_list = list(train_dataset)

    # Convert scores to float32 type to save in json
    for example in train_data_list:
        example["score"] = float(example["score"])

    with open(f'{checkpoint_path}/../scored_anchor_data_{output_notation}_weight{scoring_weight}.json', 'w') as f:
        json.dump(train_data_list, f, ensure_ascii=False, indent=4)