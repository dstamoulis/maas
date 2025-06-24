import json
import random
import argparse
import os
import re
import shutil, sys
import subprocess
from math import comb
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from openai import OpenAI
api_client = OpenAI()

from verithoughts_utils import extract_code_block, savefile, load_jsonl, rename_modules_and_instantiations, pass_at_k

import pprint

def debug_pretty_print(question_id, sample_id, result, yosys_checkresult_dict, reflection_result=None):
    """
    Pretty-print debugging information for a single sample.
    """
    pp = pprint.PrettyPrinter(indent=2, width=80)
    
    print(f"{'='*80}")
    print(f"Question ID   : {question_id}")
    print(f"Sample ID     : {sample_id}")
    print(f"{'-'*80}")
    
    print("Generated Code:")
    print(result.get('generated_code', '<none>'))
    print(f"{'-'*80}")
    
    print("Ground Truth:")
    print(result.get('ground_truth', '<none>'))
    print(f"{'-'*80}")
    
    print("Yosys Check  :")
    print(f"  Success    : {yosys_checkresult_dict.get('success')}")
    print("  Error Dict :")
    pp.pprint(yosys_checkresult_dict.get('error_dict', {}))
    print(f"{'-'*80}")
    
    if reflection_result:
        print("Reflection Result:")
        pp.pprint(reflection_result)
        print(f"{'='*80}\n")

# Function to retrieve a specific result entry by q_id and sample_id
def get_result_entry(results, q_id, sample_id):
    """
    Fetches the result dict for the given (q_id, sample_id) pair.
    Raises ValueError if not found.
    """
    for entry in results:
        if entry.get("q_id") == q_id and entry.get("sample_id") == sample_id:
            return entry
    # If we get here, no matching entry was found
    raise ValueError(f"No result found for q_id={q_id}, sample_id={sample_id}")


# Chat Completion API
def get_gpt_reflection(query, model_name="gpt-4o-mini", temperature=0.6):

    # start_time = time.time()
    messages = [
        {"role": "system", "content": "You are an RTL expert helping me as Verilog diagnosis assistant!"},
        {"role": "user", "content": query}
    ]
    response = api_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    # elapsed_time = round(time.time() - start_time, 4)
    return response.choices[0].message.content


def build_failure_diagnosis_prompt(generated_code: str, ground_truth: str, error_log: dict) -> str:
    """
    Given an error_log dict of the form:
      {
        "success": False,
        "return_codes": [...],
        "module_success_list": [...],
        "error_dict": [{"mod1": "err1"}, {"mod2": ""}, ...],
      }
    and the .V designs
        generated_code: "..."
        ground_truth: "..."
    returns a GPT-friendly prompt asking for a concise diagnosis.
    """
    # extract only the modules that actually failed
    errors = [list(d.items())[0] for d in error_log["error_dict"] if list(d.values())[0]]
    error_lines = "\n".join(f"- {mod}: {msg.strip()}" for mod, msg in errors)

    prompt = f"""
You are a Verilog expert. A recent synthesis-and-equivalence check run has failed.  \
Below are the two versions of the design and the error messages from yosys.  \

--- Generated code:\
{generated_code}

--- Ground-truth reference:

{ground_truth}

--- Yosys error messages:
{error_lines}

Please give a **brief, pointed diagnosis** of what went wrong:
1. Exactly which part of the generated design failed and why (e.g. syntax error, mismatched logic in module 'adder', etc.).  
2. How you would correct it—describe the specific fix rather than re-posting the entire module.  

Keep your explanation as concise as possible, focusing on root cause and remedy.
"""
    return prompt

def reflect_on_yosysrun(generated_code, ground_truth, error_log):

    reflection_prompt = build_failure_diagnosis_prompt(
        generated_code, ground_truth, error_log
    )    
    return get_gpt_reflection(reflection_prompt)

parser = argparse.ArgumentParser(description="Arg Parse")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B", help="HF model name")
parser.add_argument("--question_id", type=int, default=0, help="Question id")
parser.add_argument("--sample_id", type=int, default=0, help="the n-th sample for that question")
parser.add_argument("--num_samples", type=int, default=20, help="Number of samples per question")
parser.add_argument("--gpt_reflect", action="store_true", help="Get a GPT diagnosis in this call directly")
args = parser.parse_args()

model_name = args.model_name
question_id = args.question_id
sample_id = args.sample_id
gpt_reflect = args.gpt_reflect
num_samples = args.num_samples

# NO! parser.add_argument("--yosys_location", type=str, help="Absolute path to yosys environment.") JUST EXPORT IN PATH!!
# NO! yosys_location = args.yosys_location
# HECK NO! parser.add_argument("--old_data", action="store_true", help="Old data format") # WTH?

# NO! benchmark_data = load_json(args.benchmark_path)
# NO! parser.add_argument("--benchmark_path", type=str, default="VeriThoughtsBenchmark", help="Path to the benchmark jsonl")
# YES! Login using e.g. `huggingface-cli login` to access this dataset
# benchmark_data = load_dataset("wilyub/VeriThoughtsBenchmark", split="train")

# Directory: benchmark_results/{model_name}/
_names_list = [model_name, f"samples_{num_samples}"]
sub_folder = "-".join(_names_list)
results_path = os.path.join("benchmark_results", sub_folder)
os.makedirs(results_path, exist_ok=True)
results_file = os.path.join(results_path, "results.jsonl")
results_data = load_jsonl(results_file)
# Yosys evals file
yosys_evals_filename = os.path.join(results_path, "yosys_evals.jsonl")
results_file = os.path.join(results_path, "yo.jsonl")
yosys_evals_results = load_jsonl(yosys_evals_filename)


try:
    result = get_result_entry(results_data, question_id, sample_id)
    yosys_checkresult_dict = get_result_entry(yosys_evals_results, question_id, sample_id)
    # print("Found entry:", entry)
except ValueError as e:
    print("Error:", e)
    exit()

reflection_result = None
if gpt_reflect:
    if not yosys_checkresult_dict['success']:
        reflection_result = \
            reflect_on_yosysrun(result['generated_code'], result['ground_truth'], yosys_checkresult_dict)
    else:
        print("Successful run; no need to GPT-reflect")


debug_pretty_print(question_id, sample_id, result, yosys_checkresult_dict, reflection_result)