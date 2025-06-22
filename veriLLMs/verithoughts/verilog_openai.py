# dstam: 06/19/25

import sys
import os
import argparse
import math
import gc
from tqdm import tqdm
import json
import re
from copy import deepcopy
from datasets import load_dataset

from openai import OpenAI
api_client = OpenAI()

from verithoughts_utils import extract_code_block


# Chat Completion API
def gpt_get_response(query, model_name="gpt-4o-mini", temperature=0.6):

    # start_time = time.time()
    messages = [
        {"role": "system", "content": "You are an RTL expert generating Verilog code given a task description!"},
        {"role": "user", "content": query}
    ]
    response = api_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    # elapsed_time = round(time.time() - start_time, 4)
    return response.choices[0].message.content

parser = argparse.ArgumentParser(description="Arg Parse")
parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="OpenAI model name")
parser.add_argument("--num_samples_per_task", type=int, default=1, help="Number of samples per question")
parser.add_argument("--enable_reasoning", action="store_true", help="Enable if you have a reasoning mode triggered by <think>")
parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="Top p")
parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens") # Not used
parser.add_argument("--resume_gen", action="store_true", help="Enable if you want to continue from existing results.jsonl file")
args = parser.parse_args()

temperature=args.temperature
top_p = args.top_p
max_tokens=args.max_tokens
resume_gen=args.resume_gen


model_name = args.model_name
num_samples_per_task = args.num_samples_per_task
enable_reasoning = args.enable_reasoning

# NO! benchmark_data = load_json(args.benchmark_path)
# NO! parser.add_argument("--benchmark_path", type=str, default="VeriThoughtsBenchmark", help="Path to the benchmark jsonl")
# YES! Login using e.g. `huggingface-cli login` to access this dataset
benchmark_data = load_dataset("wilyub/VeriThoughtsBenchmark", split="train")

# Directory: benchmark_results/{model_name}/
results_path = os.path.join("benchmark_results", model_name)
os.makedirs(results_path, exist_ok=True)
results_file = os.path.join(results_path, "results.jsonl")
if resume_gen and os.path.exists(results_file):
    with open(results_file, "r") as rf:
        existing_results = json.load(rf)
    already_done = len(existing_results)
else:
    already_done = 0
    with open(results_file, "w") as f:
        pass # reset! their code appends indefinitely! OMG!


INSTR_SIMPLE = "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.\n" 
INSTR_REASONING = "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.<think>\n"

question_list = []
verified_benchmark_dict_list = []
for data in benchmark_data:
    if not data['verified']: continue
    for _ in range(num_samples_per_task):
        # qdata = data['question'] + INSTR_REASONING if enable_reasoning else data['question'] + INSTR_SIMPLE
        qdata = data['question'] + INSTR_SIMPLE
        question_list.append(qdata)
        verified_benchmark_dict_list.append(data)
    # break


for i, question in enumerate(tqdm(question_list, desc="Processing VeriThought questions")):
    if i < already_done:
        continue

    benchmark_dict = verified_benchmark_dict_list[i]
    question = question_list[i]
    
    gpt_response = gpt_get_response(question, model_name)
    generated_code = extract_code_block(gpt_response)
    reply_dict = {
        "question": question,
        "full_response": gpt_response,
        "generated_code": generated_code,
        "ground_truth": benchmark_dict['ground_truth']
    }
    with open(results_file, "a") as f:
        f.write(json.dumps(reply_dict) + "\n")
    # print(f"Done with idx: {i+1} / {len(question_list)}")
