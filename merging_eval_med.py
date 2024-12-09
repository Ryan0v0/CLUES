'''
Model merging and evaluation.

The benchmark evaluation components are partially adapted from OpenFedLLM 
(https://github.com/rui-ye/OpenFedLLM), specifically the MMLU evaluation routines
and data processing utilities.
'''

import os
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Dict, Optional, Sequence
import argparse
from tqdm import tqdm
from functools import partial

import pdb
from peft import (
    get_peft_model,
    PeftConfig,
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import load_dataset
import re
import time
from datetime import datetime
import json
import numpy as np
import pandas as pd

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
ISTRUCTION_DICT = {
    "medmcqa": "Given your background as a doctor, please provide your insight in addressing the medical questions based on the patient's account. Analyze the question and answer with the best option.",
    "pubmedqa": "Your role as a doctor requires you to answer the medical questions taking into account the patient's description. Analyze the question given its context. Give both long answer and yes/no decision.",
    'medqa': "You're a doctor, kindly address the medical queries according to the patient's account. Answer with the best option directly.",
    'qa2choice': "Answer this question truthfully",
    }
choices = ["A", "B", "C", "D"]
subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

def load_model(model_name_or_path, checkpoint_path):
    model = transformers.LlamaForCausalLM.from_pretrained(model_name_or_path, device_map='auto',torch_dtype=torch.float16,)

    if checkpoint_path != '':
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model.print_trainable_parameters()
    return model

def load_data(data_path):
    data = load_dataset("json", data_files=data_path, field='instances')
    data = data['train']
    return data

def construct_spedical_tokens_dict() -> dict:
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    special_tokens_dict = dict()
    if tokenizer.pad_token is None or tokenizer.pad_token == '':
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None or tokenizer.eos_token == '':
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None or tokenizer.bos_token == '':
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None or tokenizer.unk_token == '':
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    return special_tokens_dict

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def inference_on_one(input_str: Sequence[str], model, tokenizer) -> str:
    model_inputs = tokenizer(
      input_str,
      return_tensors='pt',
      padding=True,
    )

    topk_output = model.generate(
        input_ids=model_inputs.input_ids.cuda(),
        max_new_tokens=200,
        top_k=50
    )

    output_str = tokenizer.batch_decode(topk_output)  # a list containing just one str

    return output_str[0]

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1, return_token_predictions=False, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=False)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        batch_logits = model(batch_input_ids, attention_mask).logits[:, -1, :]
        if candidate_token_ids is not None:
            batch_logits = batch_logits[:, candidate_token_ids]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs

@torch.no_grad()
def eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, batch_size=1):
    prompts = []
    for i in range(0, test_df.shape[0]):
        k = args.n_mmlu_example
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        # make sure every prompt is less than 2048 tokens
        while tokenized_prompt.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            tokenized_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids

            
        prompts.append(prompt)

    answer_choice_ids = [tokenizer.encode(answer_choice, add_special_tokens=False)[0] for answer_choice in choices]
    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, return_token_predictions=False, batch_size=batch_size
    )

    # get the metrics
    cors = []
    groud_truths = test_df.iloc[:, -1].values
    predict_list = []
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)
        predict_list.append(prediction)
        
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs

def benchmark_eval(args, model, tokenizer):
    special_tokens_dict = construct_spedical_tokens_dict()
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_path_dir = {'': ''}

    model.cuda()
    acc = {}
    for key in data_path_dir.keys():
        if key == 'qa2choice':
            data = load_dataset("json", data_files=data_path_dir[key])['train']
            def simplify_result(example):
                try:
                    choice = example['output'].split('###Answer: ')[1]
                    example['output'] = choice
                except:
                    pdb.set_trace()
                    example[''] = 'A'
                return example
            data = data.map(simplify_result)
        else:
            data = load_data(data_path_dir[key])
        data = data.select(range(1000))
        output = [None] * len(data) 
        ground_truth = [None] * len(data)
        full_prompt_list = [None] * len(data)
        for i, sample in tqdm(enumerate(data), total=len(data)):
            sample['instruction'] = ISTRUCTION_DICT[key]
            full_prompt = PROMPT_DICT["prompt_input"].format_map(sample) if sample['input'] != "" else PROMPT_DICT["prompt_no_input"].format_map(sample)
            full_prompt_list[i] = full_prompt
            ground_truth[i] = sample['output']

        for i, full_prompt in tqdm(enumerate(full_prompt_list), total=len(full_prompt_list)):
            output_str = inference_on_one(full_prompt, model, tokenizer)
            output[i] = output_str

        if key == 'medmcqa' or key == 'medqa' or key == 'qa2choice':
            choice_list = []
            for out in output:
                try:
                    # response = re.findall(r'(?i)### Response: ([\s\S]*)</s>', out)[0]
                    response = out.split('### Response: ')[1]
                except:
                    print(out)
                    response = "No match found"
                try:
                    choice = re.findall(r'(?i)###Answer: OPTION\s+([A-Z])\s+IS CORRECT.', response)[0]
                except:
                    print(response)
                    choice = "No match found"

                # Check if the choice list is not empty and then convert the first element to lowercase
                if len(choice) == 0:
                    choice = "No match found"
                else:
                    choice = choice[0]
                
                choice_list.append(choice)
            
        elif key == 'pubmedqa':
            choice_list = []
            for out in output:
                try:
                    # response = re.findall(r'(?i)### Response: ([\s\S]*)</s>', out)[0]
                    response = out.split('### Response: ')[1].split('</s>')[0]
                except:
                    print(out)
                    response = "No match found"
                try:
                    choice = re.findall(r'###Answer: (yes|no|Yes|No)', response)
                except:
                    print(response)
                    choice = "No match found"

                # Check if the choice list is not empty and then convert the first element to lowercase
                if len(choice) == 0:
                    choice = "No match found"
                else:
                    choice = choice[0]
                
                choice_list.append(choice)
        # check the correctness of the output
        correct = 0
        for i in range(len(choice_list)):
            if choice_list[i].lower() == ground_truth[i].lower():
                correct += 1
        acc[key] = correct/len(choice_list)
        print("Accuracy: ", correct/len(choice_list))
    print(acc)
    return sum(acc.values()) / len(acc.values())

def mmlu_eval(args, model, tokenizer):
    
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.mmlu_data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if args.subjects:
        assert all(subj in subjects for subj in args.subjects), f"Some of the subjects you specified are not valid: {args.subjects}"
        subjects = args.subjects


    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    acc_list = []
    cor_list = []

    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):
        
        dev_df = pd.read_csv(
            os.path.join(args.mmlu_data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.n_mmlu_example]
        test_df = pd.read_csv(
            os.path.join(args.mmlu_data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval_hf_model(args, subject, model, tokenizer, dev_df, test_df)
        acc_list.append(acc)
        cor_list.append(cors)

        print(f'{subject}: {acc}')
    
    avg_acc = sum(acc_list) / len(acc_list)
    print(f'Avg acc: {avg_acc}')
    
    result = []
    for array in cor_list:
        result.extend(array)
    acc_micro = sum(result) / len(result)
    print(f'acc_micro: {acc_micro}')
    return avg_acc

def eval_gen(args, model, tokenizer):
    data = load_dataset("json", data_files=args.data_path, field='test')
    train_val = {'test': data['train']}
    val_data = train_val['test']

    results = []
    dataset_list = list(val_data)[:args.num_samples]
    with torch.no_grad():
        for point in tqdm(dataset_list):
            prompt = PROMPT_DICT['prompt_input'].format_map({"instruction":point['instruction'], 'input':point['input']})
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to("cuda"
                                            )
            generation_output = model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=True,
                    max_new_tokens=args.max_new_tokens,
                )
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            point['response'] = output.split("Response:")[1]
            results.append(point)
        
    current_time = datetime.now().strftime("%m%d_%H-%M")
    
    saved_name = args.output_dir_for_checkpoints.split('/')[-1] + '_merge_' + current_time + ".json"

    file_path = os.path.join(args.output_dir, saved_name)
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--output_dir_for_checkpoints', type=str, default='')
    parser.add_argument('--checkpoint_num', type=str, default='156')
    parser.add_argument('--mmlu_data_dir', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_token_length', type=int, default=2048)
    parser.add_argument('--n_mmlu_example', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=200, help="Number of samples to generate")
    parser.add_argument('--max_new_tokens', type=int, default=256, help="Number of new tokens to generate")
    parser.add_argument('--output_dir', type=str, default='eval_result/gen')
    parser.add_argument('--benchmark_eval', action='store_true')
    parser.add_argument('--mmlu_eval', action='store_true')
    parser.add_argument('--eval_gen', action='store_true')
    parser.add_argument('--subjects', nargs="*")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    # config = PeftConfig.from_pretrained("/nfs-share/wz341/NIPS2024/alpaca-lora/output/llama7b-multilingual/checkpoint-400")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map="auto").eval()
    model = PeftModel.from_pretrained(model, f"{args.output_dir_for_checkpoints}/Chinese/checkpoint-{args.checkpoint_num}", adapter_name="zh")
    
    _ = model.load_adapter(f"{args.output_dir_for_checkpoints}/English/checkpoint-{args.checkpoint_num}", adapter_name="en")
    _ = model.load_adapter(f"{args.output_dir_for_checkpoints}/French/checkpoint-{args.checkpoint_num}", adapter_name="fr")
    _ = model.load_adapter(f"{args.output_dir_for_checkpoints}/Japanese/checkpoint-{args.checkpoint_num}", adapter_name="jp")
    _ = model.load_adapter(f"{args.output_dir_for_checkpoints}/Russian/checkpoint-{args.checkpoint_num}", adapter_name="ru")
    _ = model.load_adapter(f"{args.output_dir_for_checkpoints}/Spanish/checkpoint-{args.checkpoint_num}", adapter_name="es")
    
    adapters = ["zh", "en", "fr", "jp", "ru", "es"]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    adapter_name = "merge"
    density = 0.2
    model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=density)

    model.set_adapter("merge")
    
    # model = load_model(args.model_name_or_path, args.checkpoint_path)
    transformers.set_seed(args.seed)
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_token_length,
        padding_side="left",
        use_fast=False,
    )
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    # if args.benchmark_eval:
    #     benchmark_avg = benchmark_eval(args, model, tokenizer)
    
    # if args.mmlu_eval:
    #     mmlu_avg = mmlu_eval(args, model, tokenizer)
    #     print(f'benchmark_avg: {benchmark_avg}, mmlu_avg: {mmlu_avg}')
    
    if args.eval_gen:
        eval_gen(args, model, tokenizer)