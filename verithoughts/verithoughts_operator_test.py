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
import asyncio

# that's for vLLM
from openai import OpenAI
api_client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)
# that's for OpenAI async
from openai import AsyncOpenAI
api_client_async = AsyncOpenAI()

from verithoughts_utils import extract_code_block, load_jsonl_file, get_result_entry, load_jsonl
from verithoughts_prompts import *
from verithoughts_operator_generate import get_vllm_response, get_openai_response, openai_reasoning_models, vllm_reasoning_models




if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Arg Parse")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--use_vllm", action="store_true", help="Enable if you want to run with vLLM")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per question")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of LLM requests to run concurrently")
    parser.add_argument("--vllm_reasoning", action="store_true", help="Enable if you have a reasoning mode triggered by <think>")
    parser.add_argument(
        "--openai_reasoning_effort",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="How much reasoning effort to spend (one of: low, medium, high)."
    ) # Following OpenAI API: https://platform.openai.com/docs/guides/reasoning?api-mode=chat#get-started-with-reasoning
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens") # Not used
    parser.add_argument("--resume_gen", action="store_true", help="Enable if you want to continue from existing results.jsonl file")
    parser.add_argument("--use_verigrad", action="store_true", help="Enable if you want to use verigrad")
    args = parser.parse_args()

    temperature=args.temperature
    top_p = args.top_p
    max_tokens=args.max_tokens
    resume_gen=args.resume_gen

    model_name = args.model_name
    num_samples = args.num_samples
    batch_size = args.batch_size

    use_verigrad = args.use_verigrad
    use_vllm = args.use_vllm
    openai_reasoning_effort = args.openai_reasoning_effort
    vllm_reasoning = args.vllm_reasoning

    # Following the MaAS naming
    prompt_op = "Test"

    # NO! benchmark_data = load_json(args.benchmark_path)
    # NO! parser.add_argument("--benchmark_path", type=str, default="VeriThoughtsBenchmark", help="Path to the benchmark jsonl")
    # YES! Login using e.g. `huggingface-cli login` to access this dataset
    benchmark_data = load_dataset("wilyub/VeriThoughtsBenchmark", split="train")

    # Directory: benchmark_results/{model_name}/
    _names_list = [model_name, f"samples_{num_samples}"]
    if vllm_reasoning and use_vllm:
        _names_list.append("reasoning")
    if any(model_name.startswith(prefix) for prefix in openai_reasoning_models) and not use_vllm:
        _names_list.append(openai_reasoning_effort)
    _names_list.append(prompt_op)
    if use_verigrad: _names_list.append("verigrad")
    sub_folder = "-".join(_names_list)
    print(sub_folder)
    results_path = os.path.join("benchmark_results", sub_folder)
    os.makedirs(results_path, exist_ok=True)
    results_file = os.path.join(results_path, "results.jsonl")
    if resume_gen and os.path.exists(results_file):
        existing_results = load_jsonl_file(results_file)
        already_done = len(existing_results) // num_samples
    else:
        already_done = 0
        with open(results_file, "w") as f:
            pass # reset! their code appends indefinitely! OMG!

    # also, load the Generate results    
    _names_list_generate = [model_name, f"samples_{num_samples}"]
    if vllm_reasoning and use_vllm:
        _names_list_generate.append("reasoning")
    if any(model_name.startswith(prefix) for prefix in openai_reasoning_models) and not use_vllm:
        _names_list_generate.append(openai_reasoning_effort)
    if use_verigrad: _names_list_generate.append("verigrad")
    sub_folder = "-".join(_names_list_generate)
    results_path_generate = os.path.join("benchmark_results", sub_folder)
    # Results file
    results_file_generate = os.path.join(results_path_generate, "results.jsonl")
    results_data_generate = load_jsonl(results_file_generate)
    # Yosys evals file (these require GT to get)
    yosys_evals_filename_generate = os.path.join(results_path_generate, "yosys_evals.jsonl")
    yosys_evals_results_generate = load_jsonl(yosys_evals_filename_generate)
    # Yosys syntax checks file (these don't require GT to get)
    yosys_syntaxchecks_filename_generate = os.path.join(results_path_generate, "yosys_syntax_checks.jsonl")
    yosys_syntaxchecks_results_generate = load_jsonl(yosys_syntaxchecks_filename_generate)


    question_list = []
    verified_benchmark_dict_list = []
    for data in benchmark_data:
        if not data['verified']: continue
        for _ in range(num_samples):
            # qdata = data['question'] + INSTR_REASONING if vllm_reasoning else data['question'] + INSTR_SIMPLE
            qdata = data['question'] # + INSTR_SIMPLE will add this later on!
            if use_verigrad: qdata+=verigrad
            question_list.append(qdata)
            verified_benchmark_dict_list.append(data)
        # break

    loop = asyncio.get_event_loop()
    num_batches = (len(question_list) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(question_list), batch_size), total=num_batches, desc="Processing VeriThought batches!!"):
        if i < already_done: continue

        questions_batch = question_list[i : i + batch_size]
        batch_runs = []
        for j, question in enumerate(questions_batch):
            idx = i + j
            q_id = idx // num_samples
            sample_id = idx % num_samples
            result_generate = get_result_entry(results_data_generate, q_id, sample_id)
            yosys_syntaxcheck_dict_generate = get_result_entry(yosys_syntaxchecks_results_generate, q_id, sample_id)
            success=yosys_syntaxcheck_dict_generate['success']
            error_log=yosys_syntaxcheck_dict_generate['error_log']

            # create question for self-reflect
            if success:
                assert 1==0
                llm_question = question + INSTR_SIMPLE
            else:
                llm_question = question + REFLECTION_ON_YOSYS_TEST_PROMPT.format(code_task=question, solution=result_generate['generated_code'], test_fail=error_log)

            if use_vllm:
                batch_runs.append(get_vllm_response(llm_question, model_name, temperature=temperature, vllm_reasoning=vllm_reasoning, skip_call=success))
            else:
                batch_runs.append(get_openai_response(llm_question, model_name, openai_reasoning_effort=openai_reasoning_effort, skip_call=success))

        llm_responses = loop.run_until_complete(asyncio.gather(*batch_runs))

        for j, llm_response in enumerate(llm_responses):
            idx = i + j
            benchmark_dict = verified_benchmark_dict_list[idx]
            question = question_list[idx]
            q_id = idx // num_samples
            sample_id = idx % num_samples
        
            # duplicate but anyway!
            result_generate = get_result_entry(results_data_generate, q_id, sample_id)
            yosys_syntaxcheck_dict_generate = get_result_entry(yosys_syntaxchecks_results_generate, q_id, sample_id)
            success=yosys_syntaxcheck_dict_generate['success']
            error_log=yosys_syntaxcheck_dict_generate['error_log']
            if success:
                assert llm_response == "Skipped"
                generated_code = result_generate['generated_code'] # reuse!
                llm_response_final = result_generate['full_response'] # reuse!
            else:
                generated_code = extract_code_block(llm_response)
                llm_response_final = llm_response

            reply_dict = {
                "q_id": q_id,
                "sample_id": sample_id,
                "full_response": llm_response_final,
                "generated_code": generated_code,
                "ground_truth": benchmark_dict['ground_truth']
            }
            with open(results_file, "a") as f:
                f.write(json.dumps(reply_dict) + "\n")
