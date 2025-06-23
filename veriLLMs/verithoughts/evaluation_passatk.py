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

from verithoughts_utils import extract_code_block, savefile, load_jsonl, rename_modules_and_instantiations, pass_at_k


def yosys_correctness_check(tmpfiles_yosys_path, generated_code, ground_truth, keep_log_stdout=False):

    # generated .v as file
    verilog_gen_file = os.path.join(tmpfiles_yosys_path, "verilog_gen.v")
    savefile(verilog_gen_file, generated_code)
    # ground-truth .v as file
    verilog_gt_file = os.path.join(tmpfiles_yosys_path, "verilog_truth.v")
    modified_module_golden, mod_module_list = rename_modules_and_instantiations(ground_truth)
    savefile(verilog_gt_file, modified_module_golden)
    # yosys script (continuously updated!)
    yosys_equivalence_check_file = os.path.join(tmpfiles_yosys_path, "equivalence_check.ys")

    yosys_returncode_list = []
    yosys_success_list = []
    yosys_checks_dict = []
    yosys_stdout_dict = []
    yosys_stderr_dict = []


    for original_module_name in mod_module_list:

        module_name = mod_module_list[original_module_name]
        yosys_equivalence_check_script = f"""
        read_verilog {verilog_gt_file}
        read_verilog {verilog_gen_file}
        prep; proc; opt; memory;
        clk2fflogic;
        miter -equiv -flatten {module_name} {original_module_name} miter
        sat -seq 50 -verify -prove trigger 0 -show-all -show-inputs -show-outputs -set-init-zero miter
        """
        savefile(yosys_equivalence_check_file, yosys_equivalence_check_script)
        
        # full_command = ["stdbuf", "-o0", "yosys", "-s", f"{yosys_equivalence_check_file}"] # DO NEED stdbuf!
        full_command = ["yosys", "-s", f"{yosys_equivalence_check_file}"]
        log_stdout = log_stderr = ""
        success = log_returncode = None

        try:
            result = subprocess.run(
                full_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120
            )
            # print(result)
            log_returncode = result.returncode
            if keep_log_stdout: log_stdout = result.stdout
            log_stderr = result.stderr
            success = (log_returncode == 0)

        except Exception as e:
            
            print(e)
            success = False
            log_returncode = -1

        yosys_returncode_list.append(log_returncode)
        yosys_success_list.append(success)
        yosys_checks_dict.append({original_module_name: success})
        if keep_log_stdout: # not used for now, so no need to keep lenghty logs!
            yosys_stdout_dict.append({original_module_name: log_stdout})
        if not success and log_stderr:
            yosys_stderr_dict.append({original_module_name: log_stderr})
        else:
            yosys_stderr_dict.append({original_module_name: ""})

    yosys_checkresult_dict = {}
    yosys_checkresult_dict['success'] = all(yosys_success_list)
    yosys_checkresult_dict['return_codes'] = yosys_returncode_list
    yosys_checkresult_dict['module_success_list'] = yosys_success_list
    yosys_checkresult_dict['error_dict'] = yosys_stderr_dict
    
    return yosys_checkresult_dict

parser = argparse.ArgumentParser(description="Arg Parse")
parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="HF model name")
parser.add_argument("--num_samples_per_task", type=int, default=1, help="Number of samples per question")
parser.add_argument("--reasoning_mode", action="store_true", help="Enable if you have a reasoning mode triggered by <think>")
args = parser.parse_args()

model_name = args.model_name
num_samples_per_task = args.num_samples_per_task
reasoning_mode = args.reasoning_mode

# 0) Pre-flight: fail fast if yosys doesn’t exist
if shutil.which("yosys") is None:
    sys.stderr.write("[ERROR] `yosys` not found on PATH — aborting! Make sure you export it!!\n")
    raise SystemExit(1)

# NO! parser.add_argument("--yosys_location", type=str, help="Absolute path to yosys environment.") JUST EXPORT IN PATH!!
# NO! yosys_location = args.yosys_location
# HECK NO! parser.add_argument("--old_data", action="store_true", help="Old data format") # WTH?

# NO! benchmark_data = load_json(args.benchmark_path)
# NO! parser.add_argument("--benchmark_path", type=str, default="VeriThoughtsBenchmark", help="Path to the benchmark jsonl")
# YES! Login using e.g. `huggingface-cli login` to access this dataset
benchmark_data = load_dataset("wilyub/VeriThoughtsBenchmark", split="train")

# Directory: benchmark_results/{model_name}/
results_path = os.path.join("benchmark_results", model_name)
os.makedirs(results_path, exist_ok=True)
results_file = os.path.join(results_path, "results.jsonl")
results_data = load_jsonl(results_file)
# Under that dir, have the tmp yosys files....
tmpfiles_yosys_path = os.path.join(results_path, "tmp")
os.makedirs(tmpfiles_yosys_path, exist_ok=True)
# Same for yosys results ... c'mon....
yosys_evals_filename = os.path.join(results_path, "yosys_evals.jsonl")
with open(yosys_evals_filename, "w") as f:
    pass # reset! their code appends indefinitely! OMG!

for result in tqdm(results_data, desc="Running yosys checks"):
    # result['generated_code'] = parsing_helper(result['full_response']) # WTH?? THE SOLUTION IS ALREADY PARSED!?!@#$
    yosys_checkresult_dict = yosys_correctness_check(tmpfiles_yosys_path, result['generated_code'], result['ground_truth'])
    with open(yosys_evals_filename, "a") as f:
        f.write(json.dumps(yosys_checkresult_dict) + "\n")

yosys_results_dict = load_jsonl(yosys_evals_filename) # Oh my!!!
correct_counts = []
for i in range(0, len(yosys_results_dict), num_samples_per_task):
    per_question_yosys_results_dict = yosys_results_dict[i:i+num_samples_per_task]
    correct_counter = sum(1 for sample in per_question_yosys_results_dict if sample['success']) # One-liner FTW!
    correct_counts.append(correct_counter)

print("pass@1:", pass_at_k(correct_counts, num_samples_per_task, 1))
print("pass@5:", pass_at_k(correct_counts, num_samples_per_task, 5))
print("pass@10:", pass_at_k(correct_counts, num_samples_per_task, 10))
