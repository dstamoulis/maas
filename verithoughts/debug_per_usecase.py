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
import uuid

from openai import OpenAI
api_client = OpenAI()

from verithoughts_utils import extract_code_block, savefile, load_jsonl, rename_modules_and_instantiations, pass_at_k, clear_verilogfile
from diagnosis_per_usecase import get_result_entry


def yosys_correctness_check_debug(tmpfiles_yosys_path, generated_code, ground_truth, keep_log_stdout=False):

    # generated .v as file
    tmp_fileid = uuid.uuid4().hex
    verilog_gen_file = os.path.join(tmpfiles_yosys_path, f"verilog_gen_{tmp_fileid}.v")
    savefile(verilog_gen_file, generated_code)
    # ground-truth .v as file
    verilog_gt_file = os.path.join(tmpfiles_yosys_path, f"verilog_truth_{tmp_fileid}.v")
    modified_module_golden, mod_module_list = rename_modules_and_instantiations(ground_truth)
    savefile(verilog_gt_file, modified_module_golden)
    # yosys script (continuously updated!)
    yosys_equivalence_check_file = os.path.join(tmpfiles_yosys_path, f"equivalence_check_{tmp_fileid}.ys")

    yosys_returncode_list = []
    yosys_success_list = []
    yosys_checks_dict = []
    yosys_stdout_dict = []
    yosys_stderr_dict = []

    for original_module_name in mod_module_list:

        module_name = mod_module_list[original_module_name]
        yosys_equivalence_check_script = f"""
        read_verilog -sv {verilog_gt_file}
        read_verilog -sv {verilog_gen_file}
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


    clear_verilogfile(verilog_gen_file)
    clear_verilogfile(verilog_gt_file)
    clear_verilogfile(yosys_equivalence_check_file)


    yosys_checkresult_dict = {}
    yosys_checkresult_dict['success'] = all(yosys_success_list)
    yosys_checkresult_dict['return_codes'] = yosys_returncode_list
    yosys_checkresult_dict['module_success_list'] = yosys_success_list
    yosys_checkresult_dict['error_dict'] = yosys_stderr_dict
    
    return yosys_checkresult_dict


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
# Under that dir, have the tmp yosys files....
tmpfiles_yosys_path = os.path.join(results_path, "tmp")
os.makedirs(tmpfiles_yosys_path, exist_ok=True)


try:
    result = get_result_entry(results_data, question_id, sample_id)
    yosys_checkresult_dict = get_result_entry(yosys_evals_results, question_id, sample_id)
    # print("Found entry:", entry)
except ValueError as e:
    print("Error:", e)
    exit()

yosys_checkresult_dict_again = yosys_correctness_check_debug(tmpfiles_yosys_path, result['generated_code'], result['ground_truth'])
print(yosys_checkresult_dict_again['error_dict'])