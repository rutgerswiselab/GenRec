import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import json
import random
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
import math

from datasets import load_dataset

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


load_8bit: bool = False,
base_model = '/common/users/jj635/llama/llama-7b/'
lora_weights = './checkpoint/movies'


"""
model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=load_8bit,
    torch_dtype=torch.float16,
    device_map='auto',
)

model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float16,
    device_map='auto',
)
"""


model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=load_8bit,
    torch_dtype=torch.float16,
    device_map={'':0},#could be 1
)
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float16,
    device_map={'':0},#could be 1
)

tokenizer = LlamaTokenizer.from_pretrained(base_model)


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f""" # noqa: E501
{data_point["instruction"]}

### input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""
    
    
def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    user_prompt = generate_prompt({**data_point, "output": ""})
    tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]  # could be sped up, probably
    return tokenized_full_prompt


def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    cutoff_len = 256
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

#with open("movie.json",'r', encoding='UTF-8') as f:
#    data = json.load(f)

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=10,
    num_beams=10,
    num_return_sequences=10,
)
data = load_dataset('./data/testset/',data_files="toys_test.json")
print(data)

hit5 = 0
hit10 = 0
ndcg5 = 0
ndcg10 = 0
total = 0
res = []

import pdb
for i, cur in tqdm(enumerate(data['train'])):
    label = cur['output']
    inputs = generate_prompt({**cur, "output": ""})
    inputs = tokenizer(inputs, return_tensors="pt")
    input_ids = inputs['input_ids'].to('cuda:0')
    #pdb.set_trace()
    res = []
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,#used to be True
            max_new_tokens=64,#used to be 128
        )

    with torch.no_grad():
        for i in range(10):
            temp = generation_output.sequences[i]
            cur = tokenizer.decode(temp,skip_special_tokens=True).split("### Response:")[1].strip()
            cur = cur.split("⁇")[0].strip()
            res.append(cur) 
    #print(label)
    #print(res)
            
    if label in res[:5]:
        hit5 += 1
        pos = res[:5].index(label)
        ndcg5 += 1.0 / (math.log(pos + 2) / math.log(2)) / 1.0
        #print(res)
        #print(label)

    if label in res:
        hit10 += 1
        pos = res.index(label)
        ndcg10 += 1.0 / (math.log(pos + 2) / math.log(2)) / 1.0
        #print(res)
        #print(label)

    total += 1
        
    if total % 100 == 0:
        print('The Hit@5 is:',hit5/total)
        print('The Hit@10 is:',hit10/total)
        print('The NDCG@5 is:',ndcg5/total)
        print('The NDCG@10 is:',ndcg10/total)
    
    
print('The Hit@5 is:',hit5/total)
print('The Hit@10 is:',hit10/total)
print('The NDCG@5 is:',ndcg5/total)
print('The NDCG@10 is:',ndcg10/total)