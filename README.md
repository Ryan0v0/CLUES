# CLUES🔍

Code for [NeurIPS 2024 Paper] [CLUES🔍: Collaborative Private-domain High-quality Data Selection for LLMs via Training Dynamics](https://clues-llmdataquality.github.io/)

## Project Overview

CLUES🔍 aims to evaluate and optimize data quality for large language models (LLMs) in the collaborative setting (including model merging and federated learning). The project includes three main steps: 
- Original model training with raw data (mixed-quality data)
- Data scoring and selection with global threshold \<CLUES🔍\>
- Collaborative model training and merging with selected data (high-quality)

## Installations
```bash
conda env create --file environment.yaml --name <your_env_name>
```

## Usage Steps

### Step 1: Fine-tune the Model with Mixed-quality Training Data
Run the following script to fine-tune the model:

```bash
sh finetune.sh
```

### Step 2: Data Scoring and Selection

Run the scoring.py script to calculate gradients and data scores, and select data based on the calculations:
```bash
sh Scoring.sh
```
- You can change "output_notation" to select gradients of different layers and submodules.

In our repo, we also implemented other data scoring baselines in our framework for easier usage and uniform comparison:
-  Instruction-Following Difficulty (IFD) Score from [From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning](https://arxiv.org/abs/2308.12032)
- Data Influence Score from [DataInf: Efficiently Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models](https://arxiv.org/abs/2310.00902)


### Step 3: Fine-tune the Model with Selected High-quality Data
Run the following script to fine-tune the model:

```bash
sh finetune_selected.sh
```
- In this step, you should use the selected datasets.

### Step 4: Merge Local Models and Evaluate the Global Model
Evaluate the model to verify the effectiveness of fine-tuning and data selection.  

We use GPT scoring to evaluate the model after collaborative learning. Run merging_eval_med.sh to generate the output of test sets:
```bash
sh merging_eval_med.sh 
```
Run eval_gpt.sh to score the answers:
```bash
sh eval_gpt.sh 
```

### Step 5: Analysis and result visualization

## Notes
- Ensure all relevant environments and dependencies are correctly configured before running the scripts.
- Modify script parameters according to the specific language.
- The scripts and parameters in each step may need adjustments based on specific requirements.
- If you have any questions related to the code or the paper, feel free to email Wanru (wz341@cam.ac.uk).

## Contributions
Contributions are welcome. Please submit issues and pull requests to improve this project.

## Acknowledgements
We would like to thank Colin Raffel, Haokun Liu, Marco Ciccone, Brian Lester, Meghdad Kurmanji and Stefanos Laskaridis for useful discussions and feedback.

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@inproceedings{
  zhao2024clues,
  title={{CLUES}: Collaborative Private-domain High-quality Data Selection for LLMs via Training Dynamics},
  author={Wanru Zhao and Hongxiang Fan and Shell Xu Hu and Wangchunshu Zhou and Nicholas Donald Lane},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024},
  }
```