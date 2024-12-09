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
from torch.utils.data import DataLoader
import pdb
import json
import numpy as np
from datasets import Dataset
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
# %%
PROMPT_DICT = {
    "prompt_full": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}\n\n### Response:\n{}"
    ),
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{}"
    )
}

def load_model(model_path, checkpoint_path):
    
    print(f"load:{model_path},{checkpoint_path}")

    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',torch_dtype=torch.float16)
    
    if checkpoint_path != '':
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
        model.print_trainable_parameters()
    return model

def tokenize(tokenizer, prompt, cutoff_len=1024, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(tokenizer, full_prompt, user_prompt, cutoff_len=2048, add_eos_token=True):
    tokenized_full_prompt = tokenize(tokenizer, full_prompt, cutoff_len, add_eos_token)

    tokenized_user_prompt = tokenize(tokenizer, user_prompt, cutoff_len, add_eos_token=add_eos_token)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    if add_eos_token:
        user_prompt_len -= 1

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ] 

    return tokenized_full_prompt
def generate_grad(model,output_notation):
    # qkv/layer/AB selection (just for example)
    if output_notation=='v':
        vectorized_grads = torch.cat([p.grad.cpu().view(-1) for n,p in model.named_parameters() if (p.grad is not None and 'v_proj' in n)]) 
    elif output_notation=='q':
        vectorized_grads = torch.cat([p.grad.cpu().view(-1) for n,p in model.named_parameters() if (p.grad is not None and 'q_proj' in n)]) 
    elif output_notation=='qv':
        vectorized_grads = torch.cat([p.grad.cpu().view(-1) for n,p in model.named_parameters() if (p.grad is not None)]) 
    elif output_notation=='A':
        vectorized_grads = torch.cat([p.grad.cpu().view(-1) for n,p in model.named_parameters() if (p.grad is not None and 'lora_A' in n)]) 
    elif output_notation=='B':
        vectorized_grads = torch.cat([p.grad.cpu().view(-1) for n,p in model.named_parameters() if (p.grad is not None and 'lora_B' in n)]) 
    elif output_notation=='layers.0':
        vectorized_grads = torch.cat([p.grad.cpu().view(-1) for n,p in model.named_parameters() if (p.grad is not None and output_notation in n)]) 
    elif output_notation=='noproj' or 'noproj_adam':
        vectorized_grads = torch.cat([p.grad.cpu().view(-1) for n,p in model.named_parameters() if (p.grad is not None)]) 
    else: # other layers
        vectorized_grads = torch.cat([p.grad.cpu().view(-1) for n,p in model.named_parameters() if (p.grad is not None and output_notation in n)]) 
    return vectorized_grads

def prepare_optimizer_state(optimizer_state, device):
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in optimizer_state.keys()])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1)
                       for n in optimizer_state.keys()])
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq

def compute_gradient(model, train_dataset, val_dataset_list, tokenizer, max_token_length=2048,checkpoint_path=None):
    optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
    adam_optimizer_state = torch.load(optimizer_path, map_location="cpu")["state"]
    avg, avg_sq = prepare_optimizer_state(adam_optimizer_state, "cpu")
    print("m/v:" , avg,avg_sq)
    tr_grad_dict = {} # per-sample gradient
    for idx, sample in enumerate(tqdm(train_dataset)):
        model.eval()
        model.zero_grad() # zeroing out gradient
        full_prompt = PROMPT_DICT['prompt_full'].format(sample['input'], sample['output'])
        input_prompt = PROMPT_DICT['prompt_input'].format(sample['input'])
        tokenized_input = generate_and_tokenize_prompt(tokenizer, full_prompt, input_prompt, max_token_length)
        input_ids = torch.tensor(tokenized_input['input_ids']).unsqueeze(0).to('cuda')
        attention_mask = torch.tensor(tokenized_input['attention_mask']).unsqueeze(0).to('cuda')
        labels = torch.tensor(tokenized_input['labels']).unsqueeze(0).to('cuda')
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        vectorized_grads = generate_grad(model,output_notation)
        if output_notation=='noproj':
            encoded_grads=vectorized_grads
        elif output_notation=='noproj_adam':
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-08
            weight_decay=0

            updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
            updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
            encoded_grads = updated_avg / torch.sqrt(updated_avg_sq + eps) + weight_decay*vectorized_grads # consider weight decay

        else:
            if 'Project' not in globals():
                print("set Projector!")
                global Project
                Project=BasicProjector(grad_dim=vectorized_grads.shape[0],proj_dim=proj_d,seed=seed, proj_type=ProjectionType.rademacher, device = device,dtype=torch.float16)
            encoded_grads=Project.project(vectorized_grads.reshape(1,-1).to('cuda'),model_id=Project.model_id)
        
        tr_grad_dict[idx] = encoded_grads
        print(f"tr_grad_dict[{idx}] = {tr_grad_dict[idx]}")

    val_grad_dict_list=[]
    for val_dataset in val_dataset_list:
        val_grad_dict = {}
        for idx, sample in enumerate(tqdm(val_dataset)):
            model.zero_grad() # zeroing out gradient
            full_prompt = PROMPT_DICT['prompt_full'].format(sample['input'], sample['output'])
            input_prompt = PROMPT_DICT['prompt_input'].format(sample['input'])
            tokenized_input = generate_and_tokenize_prompt(tokenizer, full_prompt, input_prompt, max_token_length)
            input_ids = torch.tensor(tokenized_input['input_ids']).unsqueeze(0).to('cuda')
            attention_mask = torch.tensor(tokenized_input['attention_mask']).unsqueeze(0).to('cuda')
            labels = torch.tensor(tokenized_input['labels']).unsqueeze(0).to('cuda')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            vectorized_grads = generate_grad(model,output_notation)

            if output_notation=='noproj' or 'noproj_adam':
                encoded_grads=vectorized_grads
            else:
                encoded_grads=Project.project(vectorized_grads.reshape(1,-1).to('cuda'),model_id=Project.model_id)

            val_grad_dict[idx] = encoded_grads
            print(f"val_grad_dict[{idx}] = {val_grad_dict[idx]}")
        val_grad_dict_list.append(val_grad_dict)

    return tr_grad_dict, val_grad_dict_list

def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    """ obtain gradients with adam optimizer states. """
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08

    vectorized_grads = torch.cat(
        [p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

    return vectorized_grads

# %%
# tr_lang='English'
output_notation='noproj_adam' #['noproj_adam','noproj','q','v','qv','A','B','layers.0','layers.31']

lang_list=["Chinese", "English", "French", "Japanese", "Russian", "Spanish"]
# lang_list=["Spanish"]

val_lang_list=["Chinese", "English", "French", "Japanese", "Russian", "Spanish"]

checkpoint_nums=['65','130','195','260']
model_name_or_path= '' 
max_token_length=2048
proj_d=8192
seed=42 
overwrite=False

for tr_lang in lang_list:
    checkpoint_path_list=[f'{tr_lang}/checkpoint-{num}' for num in checkpoint_nums]
    train_set_path= f'data/{tr_lang}_mixed-quality.json' 
    eval_set_path_list= [f'data/{lang}_val.json' for lang in val_lang_list]
    tr_grad_file_path_list=[f'{checkpoint_path}/{output_notation}/train_gradients.pkl' for checkpoint_path in checkpoint_path_list]
    val_grad_file_path_checkpoint_list=[[f'{checkpoint_path}/{output_notation}/val_gradients_{lang}.pkl'  for lang in val_lang_list] for checkpoint_path in checkpoint_path_list]

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

    for checkpoint_path,tr_grad_file_path, val_grad_file_path_list in zip(checkpoint_path_list, tr_grad_file_path_list,val_grad_file_path_checkpoint_list): # Loop through different checkpoints
        if os.path.exists(tr_grad_file_path) and overwrite==False:
            print(f"grad has already existed in {checkpoint_path}")
            continue
        else:
            model = load_model(model_name_or_path, checkpoint_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,model_max_length=max_token_length)
            print(f"Compute gradient for checkpoint {checkpoint_path}:")
            tr_grad_dict, val_grad_dict_list = compute_gradient(model, train_dataset, val_dataset_list, tokenizer, max_token_length=max_token_length,checkpoint_path=checkpoint_path)
            del model
            # Get directory path
            dir_path = f'{checkpoint_path}/{output_notation}'
            # Create directory if it doesn't exist
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # Save training set gradient dictionary to pickle file
            with open(tr_grad_file_path, 'wb') as f:
                pickle.dump(tr_grad_dict, f)
            # Save validation set gradient dictionary to pickle file
            for val_grad_file_path,val_grad_dict in zip(val_grad_file_path_list,val_grad_dict_list):
                with open(val_grad_file_path, 'wb') as f:
                    pickle.dump(val_grad_dict, f)
            print(f"finish grad calculation in {checkpoint_path}")
