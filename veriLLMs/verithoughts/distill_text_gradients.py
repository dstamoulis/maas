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
def get_gpt_uber_reflection(query, model_name="gpt-4o-mini", temperature=0.2):

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


parser = argparse.ArgumentParser(description="Arg Parse")
parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="HF model name")
parser.add_argument("--num_gradients", type=int, default=4, help="Number of num_gradients")
parser.add_argument("--num_samples_per_task", type=int, default=1, help="Number of samples per question")
parser.add_argument("--reasoning_mode", action="store_true", help="Enable if you have a reasoning mode triggered by <think>")
args = parser.parse_args()

num_gradients=args.num_gradients
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

# Directory: benchmark_results/{model_name}/ -- Reflections file
results_path = os.path.join("benchmark_results", model_name)
reflections_filename = os.path.join(results_path, "error_reflections.jsonl")
reflection_results = load_jsonl(reflections_filename)

diagnoses_list = []
for i, reflection_result in enumerate(tqdm(reflection_results, desc="Collecting reflections")):
    # print(reflection_result)

    _success = reflection_result["succces"]
    if not _success:
        diagnoses_list.append(reflection_result["reflection"]) 
    

all_diagnoses = '\n'.join(diagnoses_list)
uber_suggestion_prompt = f"""
You are a hardware-design best-practices advisor. You have just reviewed the following aggregated Verilog failure diagnoses:

{all_diagnoses}

Your task is to extract the top {num_gradients} recurring root-cause categories and turn each one into a standalone, self-contained promptable guideline. These guidelines should be:

- Design-agnostic (no references to specific modules or error messages)
- Explicit and actionable (not vague platitudes)
- Suitable for inclusion in a “prompt gradient” to steer an LLM away from the same mistakes

Please output exactly {num_gradients} bullet points, each a single guideline such as:

1. “…”
2. “…”
3. “…”
4. “…”
...
N. ...

Do not include any additional text or numbering beyond those {num_gradients} bullets.  

"""

uber_suggestion = get_gpt_uber_reflection(uber_suggestion_prompt)
print(uber_suggestion)


# For N=4, I got:
verigrad="""

GOOD DESIGN PRACTICES + SUGGESTIONS:

- Ensure that all output signals are assigned using continuous assignments rather than procedural assignments within always blocks to maintain correct timing and behavior in combinational logic.

- Use proper sensitivity lists in always blocks, ensuring that they only include clock edges or specific conditions relevant to the intended functionality, avoiding asynchronous resets unless explicitly required.

- Implement logic using the appropriate gate types (AND, OR, NOR, NAND) as defined in the design specifications, ensuring that the logical operations accurately reflect the intended behavior without introducing unnecessary complexity or incorrect signal relationships.

- Clearly define and handle reset conditions in a synchronous manner, ensuring that all relevant signals are initialized or cleared appropriately during reset to avoid undefined behavior during operation.

"""