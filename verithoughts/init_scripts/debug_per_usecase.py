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
from generate_allops import get_results_filepath


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


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Arg Parse")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B", help="HF model name")
    parser.add_argument("--question_id", type=int, default=0, help="Question id")
    parser.add_argument("--sample_id", type=int, default=0, help="the n-th sample for that question")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples per question")
    parser.add_argument("--gpt_reflect", action="store_true", help="Get a GPT diagnosis in this call directly")
    parser.add_argument("--vllm_reasoning", action="store_true", help="Enable if you have a reasoning mode triggered by <think>")
    parser.add_argument(
        "--openai_reasoning_effort",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="How much reasoning effort to spend (one of: low, medium, high)."
    ) # Following OpenAI API: https://platform.openai.com/docs/guides/reasoning?api-mode=chat#get-started-with-reasoning
    parser.add_argument("--use_vllm", action="store_true", help="Enable if you want to run with vLLM")
    parser.add_argument(
        "--prompt_op",
        type=str,
        choices=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Test", "SelfRefine", "MultiRefine", "EarlyStop", "ReAct", "Debate"],
        default="Generate",
        help="Which LLM prompting technique to use (CoT, Ensemble, etc.)."
    ) # Following the MaAS naming
    parser.add_argument("--verilogeval", action="store_true", help="Enable if you have the verilogeval dataset")
    parser.add_argument("--refine_op", action="store_true", help="Enable if you want to refine that op")
    parser.add_argument("--self_refine_op", action="store_true", help="Enable if you want to use refine directly at runtime")
    args = parser.parse_args()

    model_name = args.model_name
    question_id = args.question_id
    sample_id = args.sample_id
    gpt_reflect = args.gpt_reflect
    num_samples = args.num_samples
    vllm_reasoning = args.vllm_reasoning
    openai_reasoning_effort = args.openai_reasoning_effort
    
    use_vllm = args.use_vllm
    prompt_op = args.prompt_op
    verilogeval = args.verilogeval
    refine_op = args.refine_op
    self_refine_op = args.self_refine_op

    # YES! Login using e.g. `huggingface-cli login` to access this dataset
    # benchmark_data = load_dataset("wilyub/VeriThoughtsBenchmark", split="train")
    if verilogeval:
        benchmark_data = load_dataset("dakies/nvlabs-verilogeval-v2-spec-to-rtl", split="test")
        benchmark_results_dest = "benchmark_results_verilogeval"
    else:
        benchmark_data = load_dataset("wilyub/VeriThoughtsBenchmark", split="train")
        benchmark_results_dest = "benchmark_results"

    # Directory: benchmark_results/{model_name}/
    results_file, results_path =  \
        get_results_filepath(model_name, num_samples, vllm_reasoning, 
                            use_vllm, prompt_op, benchmark_results_dest,
                            refine_op, self_refine_op, openai_reasoning_effort)
    results_data = load_jsonl(results_file)
    # Yosys evals file
    yosys_evals_filename = os.path.join(results_path, "yosys_evals.jsonl")
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