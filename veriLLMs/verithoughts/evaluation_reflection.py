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
1. Exactly which part of the generated design failed and why (e.g. syntax error, mismatched logic in module ‘adder’, etc.).  
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
parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="HF model name")
parser.add_argument("--num_samples_per_task", type=int, default=1, help="Number of samples per question")
parser.add_argument("--reasoning_mode", action="store_true", help="Enable if you have a reasoning mode triggered by <think>")
args = parser.parse_args()

model_name = args.model_name
num_samples_per_task = args.num_samples_per_task
reasoning_mode = args.reasoning_mode

# NO! parser.add_argument("--yosys_location", type=str, help="Absolute path to yosys environment.") JUST EXPORT IN PATH!!
# NO! yosys_location = args.yosys_location
# HECK NO! parser.add_argument("--old_data", action="store_true", help="Old data format") # WTH?

# NO! benchmark_data = load_json(args.benchmark_path)
# NO! parser.add_argument("--benchmark_path", type=str, default="VeriThoughtsBenchmark", help="Path to the benchmark jsonl")
# YES! Login using e.g. `huggingface-cli login` to access this dataset
# benchmark_data = load_dataset("wilyub/VeriThoughtsBenchmark", split="train")

# Directory: benchmark_results/{model_name}/
results_path = os.path.join("benchmark_results", model_name)
os.makedirs(results_path, exist_ok=True)
results_file = os.path.join(results_path, "results.jsonl")
results_data = load_jsonl(results_file)
# Yosys evals file
yosys_evals_filename = os.path.join(results_path, "yosys_evals.jsonl")
results_file = os.path.join(results_path, "yo.jsonl")
yosys_evals_results = load_jsonl(yosys_evals_filename)
# Reflections file
reflections_filename = os.path.join(results_path, "error_reflections.jsonl")
with open(reflections_filename, "w") as f:
    pass # reset! their code appends indefinitely! OMG!


for i, yosys_checkresult_dict in enumerate(tqdm(yosys_evals_results, desc="Reflecting on yosys results")):

    result = results_data[i]
    _success = yosys_checkresult_dict["success"]
    reflection_result = {"succces": _success, "reflection": "N/A"}
    if not _success:
        reflection_result["reflection"] = \
            reflect_on_yosysrun(result['generated_code'], result['ground_truth'], yosys_checkresult_dict)
    with open(reflections_filename, "a") as f:
        f.write(json.dumps(reflection_result) + "\n")
