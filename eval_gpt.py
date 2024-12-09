import argparse
import json
import os
import time
import numpy as np

import openai
from tqdm import tqdm
import asyncio
from typing import Any
import logging
from typing import List, Dict, Any
import pdb
import tiktoken
import random
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import threading

def retry_with_exponential_backoff(
        func: Any,
        initial_delay: float = 5,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 100,
        errors: Any = (
                openai.error.RateLimitError, openai.error.ServiceUnavailableError,
                openai.error.APIConnectionError, openai.error.APIError, openai.error.Timeout
        ),
) -> Any:
    """A wrapper. Retrying a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        # Loop until a response or max_retries is hit or an exception is raised.
        while True:
            try:
                print('\033[91mcalling gpt...\033[0m')
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as exce:
                logging.info(exce._message)
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(exce) from exce(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                # Sleep for the delay
                time.sleep(delay)
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def call_gpt(model: str, prompt: str, sys_prompt: str) -> str:
    # is_obs: False for prior,True for simple
    """Perform a single api call with specified model and prompt."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                    },],
        temperature=0,
        # temperature=1.5,
        max_tokens=1024,
    )
    msg = response["choices"][0]["message"]
    assert msg["role"] == "assistant", "Incorrect role returned."
    ans = msg["content"]
    return ans


def parse_score(review):
    try:
        score = float(review)
        return score
    except Exception as e:
        logger.error(
            f"{e}\nContent: {review}\n" "You must manually fix the score pair."
        )
        return [-1, -1]


def gen_prompt(ques, ans1, ans2,instruction, GT=True):
    if GT:
        sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer."

        prompt_template = (
            "Instruction: \n{criteria}\n\n"
            "Input: \n{instruction} {question}\n\n"
            "Ground Truth Response: \n{answer_1}\n\n"
            "AI Assistant's Response: \n{answer_2}\n\n"
            "YOUR SCORE: \n"
        )

        criteria = (
            "Rate the response as 1 if it is accurate and matches the Ground Truth Answer. Rate it as 0 if it is inaccurate or doesn't match the Ground Truth Answer. Provide the score as a single number, either 0 or 1."
            )
        prompt = prompt_template.format(
            instruction=instruction, question=ques, answer_1=ans1, answer_2=ans2, criteria=criteria
        )

    else:

        sys_prompt = 'We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following. \n\nInstruction: [Instruction] \nInput: [Input] \nResponse: [Response]'
        prompt_template = (
            "Instruction: \n{criteria}\n\n"
            "Input: \n{instruction} {question}\n\n"
            "Response: \n {answer_2} \n\n"
            "YOUR SCORE: \n"
        )
        # criteria = 'Please rate according to the helpfulness of the response to the instruction and the input. Each assistant receives a score on a scale of 0 to 5, where a higher score indicates higher level of the helpfulness. Please output a single line containing the int value indicating the scores.'
        criteria = 'Please rate according to the helpfulness and thruthfulness of the response to the instruction and the input. Each assistant receives a score on a scale of 0 to 5, where a higher score indicates higher level of the helpfulness and accuracy. Please output a single line containing the int value indicating the scores.'

        prompt = prompt_template.format(
            instruction=instruction, question=ques, answer_1=ans1, answer_2=ans2, criteria=criteria
        )    
    return sys_prompt, prompt

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

def save_progress(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wraped_file",default='')
    parser.add_argument("--api_key",type=str,default='')
    parser.add_argument("--api_model",type=str,default='gpt-4-1106-preview')
    parser.add_argument("--api_base",type=str,default="")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to call OpenAI GPT")
    parser.add_argument("--max_tokens", type=int, default=1024, help="maximum number of tokens produced in the output")
    parser.add_argument("--eval_size", type=int, default=200)
    parser.add_argument("--GT", type=bool, default=False, help='with groundtruth')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--num_eval', type=int, default=1)
    args = parser.parse_args()

    if args.api_base != '':
        openai.api_base = args.api_base
    openai.api_key = args.api_key
    print('Begin:',args.wraped_file)

    wraped_info = json.load(open(args.wraped_file))
    dataset_name = 'fiqa'
    qa_jsons = wraped_info

    if(dataset_name=="vicuna"):
        prompt_key = 'text'
    elif dataset_name=='fiqa':
        prompt_key = 'input'

    # total_len = len(qa_jsons)
    total_len = args.eval_size
    question_idx_list = list(range(total_len))
    print('num evaluation samples:', total_len)

    predictions_all = []
    message_list = []
    token_len_list = []
    acc_list = []
    wrong_response = 0
    avg_score_list = []
    for _ in range(args.num_eval):
        for i in tqdm(question_idx_list):
            
            ques = qa_jsons[i][prompt_key]
            ans1 = qa_jsons[i]['output']
            ans2 = qa_jsons[i]['response']
            if ans2 == ' </s>':
                continue
            instruction = qa_jsons[i]['instruction']

            # remove instruction and input from response
            cleaned_ans = ans2.replace(ques, '').replace(instruction, '').replace('\n','').replace('###', '').replace('Instruction:', '').replace('Input:','').replace('   ', '')
            sys_prompt, prompt = gen_prompt(ques, ans1, cleaned_ans,instruction, GT=args.GT)
            # sys_prompt, prompt = gen_prompt(ques, ans1, ans1,instruction, GT=args.GT)
            ans = call_gpt(args.api_model, prompt, sys_prompt)
            print(f'sample: {i}; response: {ans}')
            print(prompt)

            # Update qa_jsons with the score
            try:
                qa_jsons[i]['score'] = float(ans)
                acc_list.append(float(ans))
                print("score:", float(ans))
            except:
                print(f'wrong response: {ans}')
                wrong_response += 1
                acc_list.append(0)
                qa_jsons[i]['score'] = 0
            # Optionally, save progress periodically (e.g., every 10 items)
            if i % 10 == 0 and args.GT:
                save_progress(qa_jsons, filename=args.wraped_file)

        print("acc_list:", acc_list)
        avg_scores = np.array(acc_list).mean(0)
        std_scores = np.array(acc_list).std(0)
        avg_score_list.append(avg_scores)

        
        print(f'Avg score: {avg_scores}; Std score: {std_scores}')

        if args.save_dir != '':
            hist, bins = np.histogram(acc_list, bins='auto')
            current_time = datetime.now().strftime("%m%d_%H-%M")
            saved_name = args.save_dir+args.wraped_file.split('/')[-1].split('.')[0] + '_' + current_time + ".png"
            plt.hist(acc_list, bins=bins, alpha=0.7, color='blue')
            plt.title(args.wraped_file)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.savefig(saved_name)

    print('Avg score:', avg_score_list , 'eval num:', args.num_eval)
    print('Avg score mean:', np.array(avg_score_list).mean(0))

    print('Finish:',args.wraped_file)
    print('wrong response:', wrong_response)
