from util.influence import IFEngine
import torch
import json
from datasets import Dataset
import fire
from tqdm import tqdm
import pickle
import pdb
import numpy as np

def normalize_column(data_column):
    """
    Normalize a data column to have mean 0 and standard deviation 1.
    
    Parameters:
        data_column (list or numpy.array): The data column to be normalized.
        
    Returns:
        list: The normalized data column.
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

def run(tr_lang='English',
        train_set_path='',
        score_save_path=''
):
    print(f"tr_lang={tr_lang}\n"
          f"train_set_path={train_set_path}")
    
    grad_dir='./grad_'+train_set_path.split('/')[-1][:-5]
    print(f'grad_dir={grad_dir}')

    tr_grad_file_path=f'{grad_dir}/train_gradients.pkl'
    val_grad_file_path=f'{grad_dir}/val_gradients_{tr_lang}.pkl' 

    with open(train_set_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    train_dataset = Dataset.from_list(train_data)

    # Select a checkpoint
    ckpt_scores_list=[]
    try:
        with open(tr_grad_file_path, 'rb') as file:
            tr_grad_dict = pickle.load(file) #requires cuda
        with open(val_grad_file_path, 'rb') as file:
            val_grad_dict = pickle.load(file) 
    except:
        print("lack grad files", flush=True)

    # pdb.set_trace()
    val_grad_dict={i:{"lora":val_grad_dict[i]} for i in val_grad_dict.keys()}
    tr_grad_dict={i:{"lora":tr_grad_dict[i]} for i in tr_grad_dict.keys()}

    compute_accurate=False
    influence_engine = IFEngine()
    noise_index=0
    influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict, noise_index)
    influence_engine.compute_hvps(compute_accurate=compute_accurate)
    Datainf_score=influence_engine.compute_IF()

    # pdb.set_trace()
    if 'Datainf_score' in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns('Datainf_score')
    train_dataset= train_dataset.add_column('Datainf_score', (Datainf_score))
    train_data_list = list(train_dataset)
    with open(f'{score_save_path}', 'w') as f:
        json.dump(train_data_list, f, ensure_ascii=False, indent=4)

    del tr_grad_dict, val_grad_dict, noise_index, influence_engine
    with torch.no_grad():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    fire.Fire(run)