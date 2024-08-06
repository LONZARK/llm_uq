import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import gc
# os.chdir("/home/yxo170030/data/yxo170030/projects/LLM_localUQ")
# assert os.getcwd().endswith('LLM_localUQ'), "Set current directory as project root /path/to/project/LLM_localUQ!"
### models are places under ./pretrained_LLMs
from utils.generation import generate_humaneval_prompt
from whiteboxllm import WhiteBoxLLM
# from models import WhiteBoxLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from utils.common import write_jsonl
from tqdm import tqdm

import torch
import pdb
import time
import json





def write_jsonll(file_path, data):
    with open(file_path, 'wb') as fp:
        for item in data:
            # Ensure all tensors are converted to lists
            item = {key: (value.tolist() if torch.is_tensor(value) else value) for key, value in item.items()}
            json_str = json.dumps(item) + "\n"
            fp.write(json_str.encode('utf-8'))




MODEL_LIST = [
    # 'Meta-Llama-3-8B-Instruct',
    # 'CodeQwen1.5-7B-chat',
    'Meta-Llama-3-70B-Instruct',
    'CodeLlama-70b-Instruct-hf'
]


### load HumanEval dataset
data = load_dataset("evalplus/humanevalplus")
print(f"Dataset: HumanEvalPlus loaded.")
samples = [sample for sample in data['test']]


for model_name in MODEL_LIST:
    print(f"============================== {model_name} ==============================")

    llm = WhiteBoxLLM(model_type="AutoModelForCausalLM",use_multi_gpu=True)
    model_path = os.path.join("/data/yxo170030/projects/LLM_localUQ/pretrained_LLMs", model_name)
    llm.load_pretrained(model_path)
    
    # # create saving path
    # if not os.path.exists("./saves/raw_generation/"):
    #     os.makedirs("./saves/raw_generation/")
    # save_file = "./saves/raw_generation/"+model_name+'_humaneval.jsonl'

    # create saving path
    if not os.path.exists("./saves/uq_scores/"):
        os.makedirs("./saves/uq_scores/")
    save_file = "./saves/uq_scores/"+model_name+'_humaneval.jsonl'
    
    # generate the responses
    generation_dicts = []
    for i, sample in tqdm(enumerate(samples)):
        chat = generate_humaneval_prompt(sample['prompt'])
        strat_time = time.time()
        generation_dict = llm.generate(chat, max_new_tokens=512, return_type="generation_dict")
        gen_time = time.time() - strat_time

        print('task_id', sample['task_id'])
        print('gen_time', gen_time)
        # print('generation_dict', generation_dict)
        print('new_text', generation_dict['new_text'])


        uncertainty_dict = llm.generate_uncertainty_scores(generation_dict, debug=True)
        
        generation_dicts.append({'task_id': sample['task_id'], 'uncertainty_dict': uncertainty_dict})
        if i%5 == 0:
            write_jsonll(save_file, generation_dicts)

    write_jsonll(save_file, generation_dicts)
    
    # Explicitly delete the model and clear the memory
    del llm
    del generation_dicts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Unloaded model {model_name} and cleared memory.")
    print("\n\n")