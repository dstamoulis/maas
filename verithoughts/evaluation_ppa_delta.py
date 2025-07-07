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
import time

from operators_utils import extract_code_block, savefile, load_jsonl, rename_modules_and_instantiations
from operators_utils import pass_at_k, clear_verilogfile, get_results_filepath
from yosys_utils import has_clk_signal, get_top_module
from yosys_utils import parse_power, parse_delay, get_delay, parse_area


def percent_delta(new: float, old: float) -> float:
    """
    Compute the signed percentage change from `old` to `new`.
    Returns ((new - old) / old) * 100.
    """
    return (new - old) / old * 100


def percent_delta_list(new_list, old_list) -> float:
    """
    Compute the signed percentage change from `old` to `new`.
    Returns ((new - old) / old) * 100.
    """
    if len(new_list) == 0 or len(old_list) == 0: return -1
    return percent_delta(sum(d for d in new_list)/len(new_list), sum(d for d in old_list)/len(old_list))

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
        choices=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Test", "SelfRefine", "EarlyStop",  "ReAct"],
        default="Generate",
        help="Which LLM prompting technique to use (CoT, Ensemble, etc.)."
    ) # Following the MaAS naming
    parser.add_argument("--verilogeval", action="store_true", help="Enable if you have the verilogeval dataset")
    parser.add_argument("--refine", action="store_true", help="Enable if you want to refine that op")
    parser.add_argument("--self_refine", action="store_true", help="Enable if you want to use refine directly at runtime")
    parser.add_argument('--liberty', type=str, default="skywater-pdk/libraries/sky130_fd_sc_hd/latest/timing/sky130_fd_sc_hd__tt_025C_1v80.lib", help="Liberty file to use for synthesis")
    parser.add_argument('--target_clock_period', type=int, help="Target clock period in ns", default=20)
    parser.add_argument("--ppa_op", action="store_true", help="Enable if you want to use the PPA optimize prompt")
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
    ppa_op = args.ppa_op
    if ppa_op:
        if prompt_op != "GenerateCoT":
            sys.stderr.write("[ERROR] PPA Prompting is supported as GenerateCoT variant. Relaunch with --prompt_op GenerateCoT\n")
            raise SystemExit(1)

    target_clock_period = args.target_clock_period
    liberty = args.liberty

    # 0) Pre-flight: fail fast if yosys or skywater don't exist
    if not os.path.isfile(liberty):
        sys.stderr.write(
            f"Error: Liberty file not found at:\n  {liberty}\n"
            "Please make sure you have copied there from SkyWater PDK.\n"
        )
        raise SystemExit(1)

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
    _, results_path = \
        get_results_filepath(model_name, num_samples, vllm_reasoning, 
                            use_vllm, "GenerateCoT", benchmark_results_dest,
                            refine, self_refine, openai_reasoning_effort)
    _, results_path_ppa = \
        get_results_filepath(model_name, num_samples, vllm_reasoning, 
                            use_vllm, prompt_op, benchmark_results_dest,
                            refine, self_refine, openai_reasoning_effort,
                            ppa_op=True)

    yosys_synth_results_filename = os.path.join(results_path, "yosys_ppa.jsonl")
    results_synth_data = load_jsonl(yosys_synth_results_filename)
    yosys_results_filename = os.path.join(results_path, "yosys_evals.jsonl")
    results_data = load_jsonl(yosys_results_filename)

    yosys_synth_results_filename_ppa = os.path.join(results_path_ppa, "yosys_ppa.jsonl")
    results_synth_data_ppa = load_jsonl(yosys_synth_results_filename_ppa)
    yosys_results_filename_ppa = os.path.join(results_path_ppa, "yosys_evals.jsonl")
    results_data_ppa = load_jsonl(yosys_results_filename_ppa)

    delta_area_list = []
    delta_delay_list = []
    delta_static_power_list = []
    correct_counts_list = []
    correct_counts_ppa_list = []
    idx_list = []

    area_list = []
    delay_list = []
    static_power_list = []
    area_ppa_list = []
    delay_ppa_list = []
    static_power_ppa_list = []

    correct_counts = []
    correct_counts_ppa = []
      
    # for i, result in tqdm(enumerate(results_data), total=len(results_data), desc="Eval PPA results"):  
    num_batches = (len(results_data) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(results_data), batch_size), total=num_batches, desc="Eval PPA results"):

        results_batch = results_data[i : i + batch_size]
        result_ppa_batch = results_data_ppa[i : i + batch_size]
        result_synth_batch = results_synth_data[i : i + batch_size]
        result_synth_ppa_batch = results_synth_data_ppa[i : i + batch_size]

        # if not (result_synth['yosys_success'] and result_synth_ppa['yosys_success']): continue
        # if not (result_synth['sta_success'] and result_synth_ppa['sta_success']): continue
        # if not (result['success'] and result_ppa['success']): continue
        
        # area_ppa, area = result_synth_ppa['area'], result_synth['area']
        # delay_ppa, delay = result_synth_ppa['delay'], result_synth['delay']
        # static_power_ppa, static_power = result_synth_ppa['static_power'], result_synth['static_power']

        # if not (area_ppa > 0 and area > 0): continue
        # if not (delay_ppa > 0 and delay > 0): continue
        # if not (static_power_ppa > 0 and static_power > 0): continue

        # delta_area = percent_delta(area_ppa, area)
        # delta_delay = percent_delta(delay_ppa, delay)
        # delta_static_power = percent_delta(static_power_ppa, static_power)
        
        # if delta_area < -5:
        #     delta_area_list.append(delta_area)
        #     delta_delay_list.append(delta_delay)
        #     delta_static_power_list.append(delta_static_power)
        #     correct_counts_list.append(result["success"])
        #     correct_counts_ppa_list.append(result_ppa["success"])
        #     idx_list.append(i)

        area_list_batch = []
        delay_list_batch = []
        static_power_list_batch = []
        area_ppa_list_batch = []
        delay_ppa_list_batch = []
        static_power_ppa_list_batch = []

        for j, result in enumerate(results_batch):

            idx = i + j
            q_id = idx // num_samples
            sample_id = idx % num_samples

            # result = results_batch[j]
            result_ppa = result_ppa_batch[j]
            result_synth = result_synth_batch[j]
            result_synth_ppa = result_synth_ppa_batch[j]

            area_ppa, area = result_synth_ppa['area'], result_synth['area']
            delay_ppa, delay = result_synth_ppa['delay'], result_synth['delay']
            static_power_ppa, static_power = result_synth_ppa['static_power'], result_synth['static_power']

            if result_synth['yosys_success'] and result_synth['sta_success']:
                if not (area_ppa > 0 and delay_ppa > 0 and  static_power_ppa >0): continue
                area_ppa_list_batch.append(area_ppa)
                delay_ppa_list_batch.append(delay_ppa)
                static_power_ppa_list_batch.append(static_power_ppa)

            if result_synth_ppa['yosys_success'] and result_synth_ppa['sta_success']:
                if not (area > 0 and delay > 0 and  static_power >0): continue
                area_list_batch.append(area)
                delay_list_batch.append(delay)
                static_power_list_batch.append(static_power)

        delta_area_batch = percent_delta_list(area_ppa_list_batch, area_list_batch)
        delta_delay_batch = percent_delta_list(delay_ppa_list_batch, delay_list_batch)
        delta_static_power_batch = percent_delta_list(static_power_ppa_list_batch, static_power_list_batch)

        # if not (result['success'] and result_ppa['success']): continue        
        
        if delta_area_batch < 2: 
            area_list.extend(area_list_batch)
            delay_list.extend(delay_list_batch)
            static_power_list.extend(static_power_list_batch)
            area_ppa_list.extend(area_ppa_list_batch)
            delay_ppa_list.extend(delay_ppa_list_batch)
            static_power_ppa_list.extend(static_power_ppa_list_batch)
            # correct_counts_list.append(result["success"])
            # correct_counts_ppa_list.append(result_ppa["success"])
            # idx_list.append(i)
            correct_counter = sum(1 for sample in results_batch if sample['success']) # One-liner FTW!
            correct_counts.append(correct_counter)
            correct_counter_ppa = sum(1 for sample in result_ppa_batch if sample['success']) # One-liner FTW!
            correct_counts_ppa.append(correct_counter_ppa)

    # print(f"Designs: {len(delta_area_list)}/{len(results_data)}")
    # print(f"Area: {sum(d for d in delta_area_list)/len(delta_area_list):.2f}%")
    # print(f"Delay: {sum(d for d in delta_delay_list)/len(delta_delay_list):.2f}%")
    # print(f"Static Power: {sum(d for d in delta_static_power_list)/len(delta_static_power_list):.2f}%")
    # print("-------\n    pass@1:", pass_at_k(correct_counts_list, num_samples, 1))
    # print("PPA pass@1:", pass_at_k(correct_counts_ppa_list, num_samples, 1))
    # # print(idx_list)

    print(f"Designs: PPA {len(area_ppa_list)//num_samples} -- NO PPA {len(area_list)//num_samples}")
    print(f"Area %: {percent_delta_list(area_ppa_list, area_list):.2f}%")
    print(f"Delay %: {percent_delta_list(delay_ppa_list, delay_list):.2f}%")
    print(f"Static Power %: {percent_delta_list(static_power_ppa_list, static_power_list):.2f}%")
    # print(f"Delay: {sum(d for d in delta_delay_list)/len(delta_delay_list):.2f}%")
    # print(f"Static Power: {sum(d for d in delta_static_power_list)/len(delta_static_power_list):.2f}%")
    print("-------")
    print("pass@1:", pass_at_k(correct_counts, num_samples, 1))
    print("pass@5: ", pass_at_k(correct_counts, num_samples, 5))
    print("pass@10: ", pass_at_k(correct_counts, num_samples, 10))
    print("PPA pass@1: ", pass_at_k(correct_counts_ppa, num_samples, 1))
    print("PPA pass@5: ", pass_at_k(correct_counts_ppa, num_samples, 5))
    print("PPA pass@10:", pass_at_k(correct_counts_ppa, num_samples, 10))