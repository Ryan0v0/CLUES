import torch
import json
import numpy as np
import argparse
from tqdm import tqdm
import os
import pdb
from datasets import load_dataset


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

def match_bad_data_index(origin_path, bad_path):
    
    origin_data = load_dataset("json", data_files=origin_path, field='train')['train']
    bad_data = load_dataset("json", data_files=bad_path, field='train')['train']

    # Ensure both datasets have the same length
    if len(origin_data) != len(bad_data):
        raise ValueError("Datasets must be of the same length")

    # Store indices of mismatched samples
    mismatched_indices = []

    # Iterate through datasets and compare 'output'
    for idx, (item1, item2) in enumerate(zip(origin_data, bad_data)):
        if item1['output'] != item2['output']:
            mismatched_indices.append(idx)

    return mismatched_indices

def save_selected_data(data_list, save_path, save_name):
    save_name = save_name.split('.pt')[0]
    with open(os.path.join(save_path, save_name), "w") as fw:
        json.dump(data_list, fw, indent=4)

def load_data(dataset_path):
    dataset_train = load_dataset("json", data_files=dataset_path, field='train')
    dataset_test = load_dataset("json", data_files=dataset_path, field='test')
    dataset = {'train': dataset_train['train'], 'test': dataset_test['train']}
    return dataset

def dataset_to_dict(dataset):
    # Assumes dataset is an iterable object where each element is a dictionary
    return [record for record in dataset]

def save_data_dict(dataset, save_dir, save_name):
    dataset_dict = {"train": dataset_to_dict(dataset['train']), "test": dataset_to_dict(dataset['test'])}
    json_data = json.dumps(dataset_dict, indent=4)
    save_path = os.path.join(save_dir, save_name+'.json')
    with open(save_path, "w") as file:
        file.write(json_data)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_data_path", type=str, default='')
    parser.add_argument("--json_data_path", type=str, default='')
    parser.add_argument("--json_clean_path", type=str, default='')
    parser.add_argument("--json_save_path", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str, default='')
    parser.add_argument("--checkpoint_path", type=str, default='')
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--sample_rate", type=float, default=1)
    parser.add_argument("--sample_number", type=int, default=0)
    parser.add_argument("--prompt", type=str, default='alpaca', help='wiz, alpaca')
    parser.add_argument("--save_name", type=str, default='', help='')
    args = parser.parse_args()
    return args


def filter_by_IFD(pt_data, json_data, tokenizer, max_length=1024, prompt='alpaca',):
    mean_rate_list = []
    mean_list_1 = []
    mean_list_2 = []
    for i in tqdm(range(len(pt_data))):

        pt_data_i = pt_data[i]
        loss_1_list = pt_data_i['token_loss'][1]
        loss_2_list = pt_data_i['token_loss'][2]

        json_data_i = json_data[i]
        instruct_i = json_data_i['instruction']
        output_i = json_data_i['output'] if 'output' in json_data_i.keys() else json_data_i['response']

        direct_answer_text = '### Response:' + output_i
        if prompt == 'wiz':
            whole_text = instruct_i+'\n\n### Response:'+output_i
        elif prompt == 'alpaca':
            input_i = json_data_i['input']
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

        # Tokenize the input text
        instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=max_length).to('cpu')
        instruct_i_len = instruct_i_input_ids.shape[1] 

        def get_loss_part_text(tokenizer, text, target_span, max_length, loss_list_):

            input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to('cpu')
            start_index = text.rfind(target_span)
            text_temp = text[:start_index]
            token_id_temp = tokenizer.encode(text_temp)
            start_token = len(token_id_temp) 
            end_token_real = input_ids.shape[1]

            loss_list = loss_list_[start_token-1:end_token_real-1] 

            return end_token_real - start_token , input_ids[0][start_token:end_token_real], np.array(loss_list)
        
        if max_length-instruct_i_len > 0:

            len_1, token_ids_1, loss_list_1 = get_loss_part_text(tokenizer, direct_answer_text, output_i, max_length-instruct_i_len+4, loss_1_list)
            len_2, token_ids_2, loss_list_2 = get_loss_part_text(tokenizer, whole_text, output_i, max_length, loss_2_list)

            if len_1 <= 0 or len_2 <= 0:
                mean_rate=None

            elif instruct_i_len + len_1 > max_length:
                mean_rate=None
            else:   
                try:
                    mean_1 = loss_list_1.mean()
                    mean_2 = loss_list_2.mean()
                    mean_rate = mean_2/mean_1
                except:
                    print(loss_list_1,loss_list_2)
                    mean_rate=None
        else:
            mean_rate=None

        mean_rate_list.append((mean_rate,i))
        mean_list_1.append((mean_1,i))
        mean_list_2.append((mean_2,i))

    return mean_rate_list

def filter_by_ppl_test(pt_data):
    output_ppl = []
    whole_ppl = []
    ratio_ppl = []
    gap_ppl = []
    for i in tqdm(range(len(pt_data))):
        pt_data_i = pt_data[i]
        output_ppl.append((pt_data_i['ppl'][1], i))
        whole_ppl.append((pt_data_i['ppl'][2],i))
        ratio_ppl.append((pt_data_i['ppl'][2]/pt_data_i['ppl'][1],i))
        gap_ppl.append((pt_data_i['ppl'][2]-pt_data_i['ppl'][1],i))

    return output_ppl, whole_ppl, ratio_ppl, gap_ppl

def normalize_column(data_column):
    """
    Normalize a data column to have mean 0 and standard deviation 1.
    
    Parameters:
        data_column (list or numpy.array): Data column to be normalized.
        
    Returns:
        list: Normalized data column.
    """
    # Convert data to numpy array
    data_array = np.array(data_column)

    # Calculate mean and standard deviation
    mean = np.mean(data_array)
    std = np.std(data_array)

    # Handle case where standard deviation is 0
    if std == 0:
        raise ValueError("Standard deviation is 0. Cannot normalize. Please check if the data column contains only constants.")

    # Normalize the data
    normalized_data = (data_array - mean) / std

    # Convert to list and return
    return normalized_data.tolist()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    from transformers import LlamaTokenizer, LlamaForCausalLM
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)

    pt_data = torch.load(args.pt_data_path, map_location=torch.device('cpu'))
    json_data = load_dataset("json", data_files=args.json_data_path)['train']

    ifd = filter_by_IFD(pt_data, json_data, tokenizer, max_length=args.max_length, prompt=args.prompt)

    if args.sample_number == 0:
        args.sample_number = int(len(pt_data)*args.sample_rate)

    # Perplexity
    output_ppl, whole_ppl, ratio_ppl, gap_ppl = filter_by_ppl_test(pt_data)

    train_dataset=json_data
    ppl_score=np.array([whole_ppl[i][0].detach().numpy() for i in range(len(whole_ppl))])
    if 'ppl_score' in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns('ppl_score')
    train_dataset= train_dataset.add_column('ppl_score', (ppl_score))
    if 'ifd_score' in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns('ifd_score')
    ifd_score=np.array([ifd[i][0] for i in range(len(ifd))])
    train_dataset= train_dataset.add_column('ifd_score', (ifd_score))

    ans_length=[len(train_dataset[i]['output']) for i in range(len(train_dataset))]
    if 'ans_length' in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns('ans_length')
    train_dataset= train_dataset.add_column('ans_length', (ans_length))

    train_data_list = list(train_dataset)

    with open(args.json_save_path, 'w', encoding='utf-8') as f:
        json.dump(train_data_list, f, ensure_ascii=False, indent=4)