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
def gpt_get_response(query, model_name="o4-mini", reasoning_effort="medium", top_p=0.95):

    # start_time = time.time()
    messages = [
        {"role": "system", "content": "You are an RTL expert generating Verilog code given a task description!"},
        {"role": "user", "content": query}
    ]
    response = api_client.chat.completions.create(
        model=model_name,
        reasoning_effort=reasoning_effort,
        messages=messages,
    )
    # elapsed_time = round(time.time() - start_time, 4)
    return response.choices[0].message.content

parser = argparse.ArgumentParser(description="Arg Parse")
parser.add_argument("--model_name", type=str, default="o4-mini", help="OpenAI model name")
parser.add_argument("--num_samples_per_task", type=int, default=1, help="Number of samples per question")
parser.add_argument(
    "--reasoning_effort",
    type=str,
    choices=["low", "medium", "high"],
    default="medium",
    help="How much reasoning effort to spend (one of: low, medium, high)."
) # Following OpenAI API: https://platform.openai.com/docs/guides/reasoning?api-mode=chat#get-started-with-reasoning
parser.add_argument("--top_p", type=float, default=0.95, help="Top p") #Not used
parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens") #Not used
args = parser.parse_args()

top_p = args.top_p
max_tokens=args.max_tokens

model_name = args.model_name
num_samples_per_task = args.num_samples_per_task
reasoning_effort = args.reasoning_effort

# NO! benchmark_data = load_json(args.benchmark_path)
# NO! parser.add_argument("--benchmark_path", type=str, default="VeriThoughtsBenchmark", help="Path to the benchmark jsonl")
# YES! Login using e.g. `huggingface-cli login` to access this dataset
benchmark_data = load_dataset("wilyub/VeriThoughtsBenchmark", split="train")

# Directory: benchmark_results/{model_name}/
results_path = os.path.join("benchmark_results", model_name)
os.makedirs(results_path, exist_ok=True)
results_file = os.path.join(results_path, "results.jsonl")
with open(results_file, "w") as f:
    pass # reset! their code appends indefinitely! OMG!

INSTR_SIMPLE = "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.\n" 
INSTR_REASONING = "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.<think>\n"

question_list = []
verified_benchmark_dict_list = []
for data in benchmark_data:
    if not data['verified']: continue
    for _ in range(num_samples_per_task):
        # qdata = data['question'] + INSTR_REASONING if reasoning_effort else data['question'] + INSTR_SIMPLE # NOT used!
        qdata = data['question'] + INSTR_SIMPLE
        question_list.append(qdata)
        verified_benchmark_dict_list.append(data)
    # break


for i, question in enumerate(tqdm(question_list, desc="Processing VeriThought questions")):

    benchmark_dict = verified_benchmark_dict_list[i]
    question = question_list[i]
    
    gpt_response = gpt_get_response(question, model_name, reasoning_effort)
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
