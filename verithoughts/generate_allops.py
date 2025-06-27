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

from verithoughts_utils import extract_code_block, load_jsonl_file, load_jsonl, get_result_entry
from verithoughts_prompts import *

# OpenAI-compatible API service with vLLM
vllm_reasoning_models = ['Qwen/Qwen3'] # hardcoded!
async def get_vllm_response(query, model_name="Qwen/Qwen2.5-7B", temperature=0.6, vllm_reasoning=False, skip_call=False):

    if skip_call: return "Skipped"

    # start_time = time.time()
    messages = [
        {"role": "system", "content": "You are an RTL expert generating Verilog code given a task description!"},
        {"role": "user", "content": query}
    ]
    _extra_body_params={
        "top_k": 20,
    }
    if not vllm_reasoning and any(model_name.startswith(prefix) for prefix in vllm_reasoning_models):
        _extra_body_params["chat_template_kwargs"]= {"enable_thinking": False}
    loop = asyncio.get_running_loop() # non-blocking!
    response = await loop.run_in_executor(
        None,
        lambda: api_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=32768,
            temperature=temperature,
            top_p=0.95,
            extra_body=_extra_body_params,
        )
    )
    # elapsed_time = round(time.time() - start_time, 4)
    return response.choices[0].message.content

# Chat Completion API
openai_reasoning_models = ['o4', 'o4-mini', 'o3', 'o3-mini'] # hardcoded!
async def get_openai_response(query, model_name="gpt-4o-mini", temperature=0.6, openai_reasoning_effort="medium", top_p=0.95, skip_call=False):
    
    if skip_call: return "Skipped"

    # start_time = time.time()
    messages = [
        {"role": "system", "content": "You are an RTL expert generating Verilog code given a task description!"},
        {"role": "user", "content": query}
    ]
    try:
        if any(model_name.startswith(prefix) for prefix in openai_reasoning_models):
            response = await api_client_async.chat.completions.create(
                model=model_name,
                reasoning_effort=openai_reasoning_effort,
                messages=messages,
            )
        else:
            response = await api_client_async.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )

        reply = response.choices[0].message.content
    except Exception as e:
        print(e)
        reply = e
    # elapsed_time = round(time.time() - start_time, 4)
    return reply 


def get_benchmark_lists(benchmark_data, num_samples, verilogeval=False):
    question_list = []
    verified_benchmark_dict_list = []
    for data in benchmark_data:
        if verilogeval:
            for _ in range(num_samples):
                qdata = data['prompt']
                question_list.append(qdata)
                _data = data
                _data['ground_truth'] = _data['ref'].replace("module RefModule ", "module TopModule ")
                verified_benchmark_dict_list.append(_data)
        else:
            if not data['verified']: continue
            for _ in range(num_samples):
                qdata = data['question'] 
                question_list.append(qdata)
                verified_benchmark_dict_list.append(data)

    return question_list, verified_benchmark_dict_list


def get_results_filepath(model_name, num_samples, vllm_reasoning, use_vllm, prompt_op, benchmark_results_dest, refine_op=False):
    _names_list = [model_name, f"samples_{num_samples}"]
    if vllm_reasoning and use_vllm:
        _names_list.append("reasoning")
    if any(model_name.startswith(prefix) for prefix in openai_reasoning_models) and not use_vllm:
        _names_list.append(openai_reasoning_effort)
    _names_list.append(prompt_op)
    if refine_op: _names_list.append("Refine")
    sub_folder = "-".join(_names_list)
    results_path = os.path.join(benchmark_results_dest, sub_folder)
    os.makedirs(results_path, exist_ok=True)
    results_file = os.path.join(results_path, "results.jsonl")
    return results_file, results_path


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
    parser.add_argument("--verilogeval", action="store_true", help="Enable if you have the verilogeval dataset")
    parser.add_argument(
        "--prompt_op",
        type=str,
        choices=["Generate", "GenerateCoT", "ReAct", "ReActSimple"],
        default="Generate",
        help="Which LLM prompting technique to use (CoT, Ensemble, etc.)."
    ) # Following the MaAS naming
    parser.add_argument("--refine_op", action="store_true", help="Enable if you want to refine that op")
    args = parser.parse_args()

    temperature=args.temperature
    top_p = args.top_p
    max_tokens=args.max_tokens
    resume_gen=args.resume_gen

    model_name = args.model_name
    num_samples = args.num_samples
    batch_size = args.batch_size
    verilogeval = args.verilogeval
    prompt_op = args.prompt_op
    refine_op = args.refine_op

    use_verigrad = args.use_verigrad
    use_vllm = args.use_vllm
    openai_reasoning_effort = args.openai_reasoning_effort
    vllm_reasoning = args.vllm_reasoning

    # NO! benchmark_data = load_json(args.benchmark_path)
    # NO! parser.add_argument("--benchmark_path", type=str, default="VeriThoughtsBenchmark", help="Path to the benchmark jsonl")
    # YES! Login using e.g. `huggingface-cli login` to access this dataset
    if verilogeval:
        benchmark_data = load_dataset("dakies/nvlabs-verilogeval-v2-spec-to-rtl", split="test")
        benchmark_results_dest = "benchmark_results_verilogeval"
    else:
        benchmark_data = load_dataset("wilyub/VeriThoughtsBenchmark", split="train")
        benchmark_results_dest = "benchmark_results"

    # Destination Directory: benchmark_results/{....}/
    results_file, results_path =\
        get_results_filepath(model_name, num_samples, vllm_reasoning, 
                            use_vllm, prompt_op, benchmark_results_dest,
                            refine_op)

    if resume_gen and os.path.exists(results_file):
        existing_results = load_jsonl_file(results_file)
        already_done = len(existing_results) # // num_samples: will match since in dir-name!
    else:
        already_done = 0
        with open(results_file, "w") as f:
            pass # reset! their code appends indefinitely! OMG!

    # Source (for refine if set!)
    source_yosys_syntaxchecks_filename = None
    source_yosys_syntaxchecks_results = None
    source_results_file = source_results_path = None
    if refine_op:
        source_results_file, source_results_path =\
            get_results_filepath(model_name, num_samples, vllm_reasoning, 
                                use_vllm, prompt_op, benchmark_results_dest)
        source_results = load_jsonl_file(source_results_file)
        source_yosys_syntaxchecks_filename = os.path.join(source_results_path, "yosys_syntax_checks.jsonl")
        if not os.path.exists(source_yosys_syntaxchecks_filename): 
            sys.stderr.write("[ERROR] `yosys_syntax_checks` not found! Make sure to run evaluation_syntax_check first!!\n")
            raise SystemExit(1)

        source_yosys_syntaxchecks_results = load_jsonl(source_yosys_syntaxchecks_filename)

    question_list, verified_benchmark_dict_list= \
        get_benchmark_lists(benchmark_data, num_samples, verilogeval)

    loop = asyncio.get_event_loop()
    num_batches = (len(question_list) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(question_list), batch_size), total=num_batches, desc="Processing batches!!"):
        if i < already_done: continue

        questions_batch = question_list[i : i + batch_size]
        questions_batch_prompts = []
        questions_skip = []
        for j, q in enumerate(questions_batch):
            if prompt_op == "Generate":
                q_prompt = q + GENERATE_PROMPT
            elif prompt_op == "GenerateCoT":
                q_prompt = GENERATE_COT_PROMPT.format(code_task=q)
            elif prompt_op == "ReAct":
                q_prompt = REACT_PROMPT.format(code_task=q)
            elif prompt_op == "ReActSimple":
                q_prompt = REACT_PROMPT_SIMPLE.format(code_task=q)
            
            q_skip = False
            if refine_op:
                idx = i + j
                q_id = idx // num_samples
                sample_id = idx % num_samples
                source_result = get_result_entry(source_results, q_id, sample_id)
                source_yosys_syntaxcheck = get_result_entry(source_yosys_syntaxchecks_results, q_id, sample_id)
                success=source_yosys_syntaxcheck['success']
                error_log=source_yosys_syntaxcheck['error_log']
                if not success:
                    q_prompt = REFINE_PROMPT.format(code_task=q, solution=source_result['generated_code'], test_fail=error_log)
                else:
                    q_skip = True

            questions_skip.append(q_skip)
            questions_batch_prompts.append(q_prompt)

        if use_vllm:
            batch_runs = [
                get_vllm_response(q, model_name, temperature=temperature, vllm_reasoning=vllm_reasoning, skip_call=questions_skip[j]) 
                for j, q in enumerate(questions_batch_prompts)
            ]
        else:
            batch_runs = [
                get_openai_response(q, model_name, openai_reasoning_effort=openai_reasoning_effort, skip_call=questions_skip[j]) 
                for j, q in enumerate(questions_batch_prompts)
            ]
        llm_responses = loop.run_until_complete(asyncio.gather(*batch_runs))

        for j, llm_response in enumerate(llm_responses):
            idx = i + j
            benchmark_dict = verified_benchmark_dict_list[idx]
            question = question_list[idx]
            q_id = idx // num_samples
            sample_id = idx % num_samples
        
            if refine_op:
                source_result = get_result_entry(source_results, q_id, sample_id)
                source_yosys_syntaxcheck = get_result_entry(source_yosys_syntaxchecks_results, q_id, sample_id)
                success=source_yosys_syntaxcheck['success']
                error_log=source_yosys_syntaxcheck['error_log']
                if success:
                    assert llm_response == "Skipped"
                    generated_code = source_result['generated_code'] # reuse!
                    llm_response_final = source_result['full_response'] # reuse!
                else:
                    generated_code = extract_code_block(llm_response)
                    llm_response_final = llm_response
            else:
                generated_code = extract_code_block(llm_response)
                llm_response_final = llm_response
            
            generated_code = extract_code_block(llm_response)
            reply_dict = {
                "q_id": q_id,
                "sample_id": sample_id,
                "full_response": llm_response_final,
                "generated_code": generated_code,
                "ground_truth": benchmark_dict['ground_truth']
            }
            with open(results_file, "a") as f:
                f.write(json.dumps(reply_dict) + "\n")
