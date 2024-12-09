import os
import json
import torch
import argparse
from tqdm import tqdm

import torch.nn as nn
log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction='none')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
import transformers
from peft import PeftModel
from datasets import load_dataset
import pdb

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default='')
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default='wiz', help='wiz, alpaca')
    parser.add_argument("--mod", type=str, default='pre', help='pre, cherry')
    args = parser.parse_args()
    return args

# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad(): 
        outputs = model(input_ids, labels=input_ids.contiguous())
    loss = outputs.loss
    perplexity = torch.exp(loss)

    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1]
    sentence_embedding = embeddings.mean(dim=1)

    return perplexity.to('cpu'), sentence_embedding.to('cpu')

# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    start_index = text.rfind(target_span)
    start_token = len(tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]

    labels = input_ids.clone()
    labels[0, :start_token] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    loss = outputs.loss
    perplexity = torch.exp(loss)

    losses = []
    logits = outputs.logits
    for i in range(1, end_token):
        log_prob_dist = log_softmax(logits[0, i-1])
        true_token = input_ids[0, i]
        token_loss = nll_loss(log_prob_dist.unsqueeze(0), true_token.unsqueeze(0))
        losses.append(token_loss.item())

    return perplexity.to('cpu'), 0, losses

def load_model(model_name_or_path, checkpoint_path):
    model = transformers.LlamaForCausalLM.from_pretrained(model_name_or_path, device_map='auto',torch_dtype=torch.float16,)

    if checkpoint_path != '':
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model.print_trainable_parameters()
    return model

def load_data(data_path):
    data = load_dataset("json", data_files=data_path)
    data = data['train']
    return data

def calculate_loss(data, tokenizer, model, prompt='alpaca', max_length=1024, mod='cherry'):
    model.eval()
    new_data = []
    for i, data_i in tqdm(enumerate(data)):
        # data_i = sampled_data[i]
        instruct_i = data_i['instruction']
        output_i = data_i['output'] if 'output' in data_i.keys() else data_i['response']

        direct_answer_text = '### Response:' + output_i
        if prompt == 'wiz':
            whole_text = instruct_i+'\n\n### Response:'+output_i
            input_i = data_i['input'] if 'input' in data_i.keys() else ''
            if input_i != '':
                whole_text = instruct_i+'\nInput:'+input_i+'\n\n### Response:'+output_i

        elif prompt == 'alpaca':
            input_i = data_i['input'] if 'input' in data_i.keys() else ''
            if input_i == '':
                temp_dict = {'instruction':instruct_i}
                promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use
            else:
                temp_dict = {'instruction':instruct_i,'input':input_i}
                promt_to_use = PROMPT_DICT["prompt_input"].format_map(temp_dict)
                whole_text = promt_to_use + output_i
                instruct_i = promt_to_use
            

        temp_data_i = {}
        if mod == 'pre':
            ppl_ins_alone, emb_ins_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, instruct_i, max_length)
            temp_data_i['ppl'] = [ppl_ins_alone,0,0]
            temp_data_i['sent_emb'] = [emb_ins_alone,0,0]

        elif mod == 'cherry':
            instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=max_length).to('cuda')
            instruct_i_len = instruct_i_input_ids.shape[1] 
        
            ppl_out_alone, _, loss_list_alone = get_perplexity_and_embedding_part_text(tokenizer, model, direct_answer_text, output_i, max_length-instruct_i_len+4)
            ppl_out_condition, _, loss_list_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, max_length)

            temp_data_i['ppl'] = [0,ppl_out_alone,ppl_out_condition]
            temp_data_i['token_loss'] = [[],loss_list_alone,loss_list_condition]

        new_data.append(temp_data_i)
        pass
    return new_data

if __name__ == "__main__":
    args = parse_args()
    print(args)

    model = load_model(args.model_name_or_path, args.checkpoint_path)
    tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_name_or_path)

    data = load_data(args.data_path)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data.select(range(start_idx, end_idx))

    import time
    strat_time = time.time()
    new_data = calculate_loss(sampled_data, tokenizer, model, args.prompt, args.max_length, args.mod)

    print('New data len:', len(new_data))
    save_name = args.save_name
    dir_name = os.path.dirname(save_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    torch.save(new_data, save_name)

    print('Time Used:',(time.time()-strat_time)/60,'(min)')