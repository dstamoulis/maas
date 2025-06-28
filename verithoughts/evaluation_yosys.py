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
import asyncio

from operators_utils import extract_code_block, savefile, load_jsonl, rename_modules_and_instantiations
from operators_utils import pass_at_k, clear_verilogfile, get_results_filepath


async def yosys_syntax_check(tmpfiles_yosys_path, generated_code, keep_log_stdout=False, skip_call=False):

    if skip_call:
        yosys_checkresult_dict = {}
        yosys_checkresult_dict['success'] = True
        yosys_checkresult_dict['return_code'] = 0
        yosys_checkresult_dict['error_log'] = "Skipped"
        return yosys_checkresult_dict

    loop = asyncio.get_running_loop()

    # generated .v as file
    tmp_fileid = uuid.uuid4().hex
    verilog_gen_file = os.path.join(tmpfiles_yosys_path, f"verilog_gen_syntax_{tmp_fileid}.v")
    savefile(verilog_gen_file, generated_code)

    # build the inline yosys script
    yosys_script = (
        f"read_verilog -sv {verilog_gen_file}; "
        "hierarchy -check; "
        "proc; "
        "opt; "
        "stat"
    )
    
    full_command = ["yosys", "-p", f"{yosys_script}"]
    log_stdout = log_stderr = ""
    success = log_returncode = None

    # Offload to thread executor
    try:
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                full_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120
            )
        )
        log_returncode = result.returncode
        if keep_log_stdout: log_stdout = result.stdout
        log_stderr = result.stderr
        success = (log_returncode == 0)
    except subprocess.TimeoutExpired as e:
        log_returncode = -1
        log_stderr = f"TimeoutExpired: {e}"
        success = False
    except Exception as e:
        log_returncode = -1
        log_stderr = f"Yosys failed: {e}"
        success = False

    clear_verilogfile(verilog_gen_file)

    yosys_checkresult_dict = {}
    yosys_checkresult_dict['success'] = success
    yosys_checkresult_dict['return_code'] = log_returncode
    yosys_checkresult_dict['error_log'] = log_stderr
    
    return yosys_checkresult_dict



async def yosys_correctness_check(tmpfiles_yosys_path, generated_code, ground_truth, keep_log_stdout=False):

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

    loop = asyncio.get_running_loop()

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

        # Offload to thread executor
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    full_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=120
                )
            )
            log_returncode = result.returncode
            if keep_log_stdout: log_stdout = result.stdout
            log_stderr = result.stderr
            success = (log_returncode == 0)
        except subprocess.TimeoutExpired as e:
            log_returncode = -1
            log_stderr = f"TimeoutExpired: {e}"
            success = False
        except Exception as e:
            log_returncode = -1
            log_stderr = f"Yosys failed: {e}"
            success = False

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
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="HF model name")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per question")
    parser.add_argument("--use_vllm", action="store_true", help="Enable if you want to run with vLLM")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of yosys runs to run concurrently")
    parser.add_argument("--vllm_reasoning", action="store_true", help="Enable if you have a reasoning mode triggered by <think>")
    parser.add_argument(
        "--openai_reasoning_effort",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="How much reasoning effort to spend (one of: low, medium, high)."
    ) # Following OpenAI API: https://platform.openai.com/docs/guides/reasoning?api-mode=chat#get-started-with-reasoning
    parser.add_argument(
        "--prompt_op",
        type=str,
        choices=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Test", "SelfRefine", "MultiRefine", "EarlyStop", "ReAct", "Debate"],
        default="Generate",
        help="Which LLM prompting technique to use (CoT, Ensemble, etc.)."
    ) # Following the MaAS naming
    parser.add_argument("--verilogeval", action="store_true", help="Enable if you have the verilogeval dataset")
    parser.add_argument("--refine", action="store_true", help="Enable if you want to refine that op")
    parser.add_argument("--self_refine", action="store_true", help="Enable if you want to use refine directly at runtime")
    parser.add_argument("--syntax_check", action="store_true", help="Enable if you want to just check syntax w/o GTs")
    parser.add_argument("--cost_metrics", action="store_true", help="Enable if you want to just get LLM cost metrics w/o yosys")
    args = parser.parse_args()

    model_name = args.model_name
    num_samples = args.num_samples
    vllm_reasoning = args.vllm_reasoning
    openai_reasoning_effort = args.openai_reasoning_effort

    use_vllm = args.use_vllm
    batch_size = args.batch_size
    prompt_op = args.prompt_op
    verilogeval = args.verilogeval
    refine = args.refine
    self_refine = args.self_refine
    syntax_check = args.syntax_check
    cost_metrics = args.cost_metrics

    if verilogeval: batch_size = 10 # 1GB yosys runs! wow!
    
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
    if verilogeval:
        benchmark_data = load_dataset("dakies/nvlabs-verilogeval-v2-spec-to-rtl", split="test")
        benchmark_results_dest = "benchmark_results_verilogeval"
    else:
        benchmark_data = load_dataset("wilyub/VeriThoughtsBenchmark", split="train")
        benchmark_results_dest = "benchmark_results"

    # Directory: benchmark_results/{model_name}/
    results_file, results_path = \
        get_results_filepath(model_name, num_samples, vllm_reasoning, 
                            use_vllm, prompt_op, benchmark_results_dest,
                            refine, self_refine, openai_reasoning_effort)
    results_data = load_jsonl(results_file)
    # Under that dir, have the tmp yosys files....
    tmpfiles_yosys_path = os.path.join(results_path, "tmp")
    os.makedirs(tmpfiles_yosys_path, exist_ok=True)
    # Same for yosys results ... c'mon....
    yosys_evals_basename = "yosys_syntax_checks.jsonl" if syntax_check else "yosys_evals.jsonl"
    yosys_evals_filename = os.path.join(results_path, yosys_evals_basename)
    with open(yosys_evals_filename, "w") as f:
        pass # reset! their code appends indefinitely! OMG!


    if cost_metrics:

        _print_msg = "Model: LLM cost averages"
        print(_print_msg, os.path.basename(results_path))
        print('\n'.join(
            f"{label}: {sum(d[key] for d in results_data)/len(results_data):.2f}"
            for label, key in [
                ("Time elapsed",      "time_elapsed"),
                ("Prompt tokens",     "prompt_tokens"),
                ("Completion tokens", "completion_tokens"),
                ("Total tokens",      "total_tokens"),
            ]
        ))
        # for result_datapoint in results_data:
        #     elapsed_time = result_datapoint['time_elapsed']
        #     prompt_tokens = result_datapoint['prompt_tokens']
        #     completion_tokens = result_datapoint['completion_tokens']
        #     total_tokens = result_datapoint['total_tokens']
        sys.exit()


    loop = asyncio.get_event_loop()
    num_batches = (len(results_data) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(results_data), batch_size), total=num_batches, desc="Running yosys checks"):

        results_batch = results_data[i : i + batch_size]
        if syntax_check: 
            batch_runs = [
                yosys_syntax_check(tmpfiles_yosys_path, result['generated_code']) 
                for result in results_batch
            ]
        else:
            batch_runs = [
                yosys_correctness_check(tmpfiles_yosys_path, result['generated_code'], result['ground_truth']) 
                for result in results_batch
            ]

        yosys_checkresults = loop.run_until_complete(asyncio.gather(*batch_runs))

        for j, yosys_checkresult in enumerate(yosys_checkresults):
            idx = i + j
            q_id = idx // num_samples
            sample_id = idx % num_samples
            if syntax_check: 
                yosys_checkresult_dict = {
                    "q_id": q_id,
                    "sample_id": sample_id,
                    "success": yosys_checkresult["success"],
                    "return_code": yosys_checkresult["return_code"],
                    "error_log": yosys_checkresult["error_log"],
                }
            else:
                yosys_checkresult_dict = {
                    "q_id": q_id,
                    "sample_id": sample_id,
                    "success": yosys_checkresult["success"],
                    "return_codes": yosys_checkresult["return_codes"],
                    "module_success_list": yosys_checkresult["module_success_list"],
                    "error_dict": yosys_checkresult["error_dict"],
                }

            with open(yosys_evals_filename, "a") as f:
                f.write(json.dumps(yosys_checkresult_dict) + "\n")

    yosys_results_dict = load_jsonl(yosys_evals_filename) # Oh my!!!
    correct_counts = []
    for i in range(0, len(yosys_results_dict), num_samples):
        per_question_yosys_results_dict = yosys_results_dict[i:i+num_samples]
        correct_counter = sum(1 for sample in per_question_yosys_results_dict if sample['success']) # One-liner FTW!
        correct_counts.append(correct_counter)

    _print_msg = "Model: Syntax check results:" if syntax_check else "Model: Formal Verification GT results:" 
    print(_print_msg, os.path.basename(results_path))
    if num_samples >= 1:  print("pass@1:", pass_at_k(correct_counts, num_samples, 1))
    if num_samples >= 5:  print("pass@5:", pass_at_k(correct_counts, num_samples, 5))
    if num_samples >= 10: print("pass@10:", pass_at_k(correct_counts, num_samples, 10))
