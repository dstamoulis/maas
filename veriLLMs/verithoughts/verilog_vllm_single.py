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
# Set OpenAI's API key and API base to use vLLM's API server.
# Requires: pip3 install vllm==v0.9.0! https://github.com/vllm-project/vllm/issues/19432
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

api_client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

from verithoughts_utils import extract_code_block

reasoning_models = ['Qwen/Qwen3'] # hardcoded!

# OpenAI-compatible API service with vLLM
def vllm_get_response(query, model_name="Qwen/Qwen2.5-7B", temperature=0.6, enable_reasoning=False):

    # start_time = time.time()
    messages = [
        {"role": "system", "content": "You are an RTL expert generating Verilog code given a task description!"},
        {"role": "user", "content": query}
    ]
    _extra_body_params={
        "top_k": 20,
    }
    if not enable_reasoning and any(model_name.startswith(prefix) for prefix in reasoning_models):
        _extra_body_params["chat_template_kwargs"]= {"enable_thinking": False}
    response = api_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        extra_body = _extra_body_params,
    )
    # elapsed_time = round(time.time() - start_time, 4)
    return response.choices[0].message.content

parser = argparse.ArgumentParser(description="Arg Parse")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B", help="HF model name")
parser.add_argument("--num_samples_per_task", type=int, default=1, help="Number of samples per question")
parser.add_argument("--enable_reasoning", action="store_true", help="Enable if you have a reasoning mode triggered by <think>")
parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="Top p")
parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens")
args = parser.parse_args()

# based on what's hardcoded in verilog_vllm!
temperature=0.6
top_p = 0.95
max_tokens=16384

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

    benchmark_dict = verified_benchmark_dict_list[i]
    question = question_list[i]
    
    gpt_response = vllm_get_response(question, model_name, enable_reasoning)
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
