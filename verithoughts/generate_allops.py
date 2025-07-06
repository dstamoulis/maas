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
import copy
import shutil
import time

# that's for vLLM
from openai import OpenAI
api_client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)
# that's for OpenAI async
from openai import AsyncOpenAI
from openai import OpenAIError  # catch exception OpenAI client (some prompts complaint about policy)
api_client_async = AsyncOpenAI()

from operators_utils import extract_code_block, load_jsonl_file, load_jsonl, get_result_entry, get_results_filepath
from operators_prompts import *
from evaluation_yosys import yosys_syntax_check

# OpenAI-compatible API service with vLLM
vllm_reasoning_models = ['Qwen/Qwen3'] # hardcoded!
async def get_vllm_response(
        query, 
        model_name="Qwen/Qwen2.5-7B", 
        temperature=0.6, 
        vllm_reasoning=False, 
        skip_call=False, 
    ):

    if skip_call: 
        response_dict={
            'response': "Skipped",
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'time_elapsed': 0.0,
        }
        return response_dict

    # yosys_check_fail = True
    # while yosys_check_fail:

    start_time = time.time()
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
    elapsed_time = round(time.time() - start_time, 4)
    response_dict={
        'response': response.choices[0].message.content,
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens,
        'time_elapsed': elapsed_time,
    }

    return response_dict

# Chat Completion API
openai_reasoning_models = ['o4', 'o4-mini', 'o3', 'o3-mini'] # hardcoded!
async def get_openai_response(
        query, 
        model_name="gpt-4o-mini", 
        temperature=0.6, 
        openai_reasoning_effort="medium", 
        top_p=0.95, 
        skip_call=False,
    ):
    
    if skip_call: 
        response_dict={
            'response': "Skipped",
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'time_elapsed': 0.0,
        }
        return response_dict

    start_time = time.time()
    prompt_tokens= 0
    completion_tokens= 0
    total_tokens= 0

    start_time = time.time()
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
        prompt_tokens= response.usage.prompt_tokens
        completion_tokens= response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        elapsed_time = round(time.time() - start_time, 4)
        response_dict={
            'response': reply,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'time_elapsed': elapsed_time,
        }
        return response_dict 

    except OpenAIError as e:

        # this will catch InvalidRequestError (400) and other API errors
        reply = f"OpenAIError ({getattr(e, 'code', '')}): {e}"
        print(reply)
        elapsed_time = round(time.time() - start_time, 4)
        response_dict={
            'response': reply,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'time_elapsed': elapsed_time,
        }
        return response_dict 

    except Exception as e:
        reply = f"Calling OpenAI errorred out: {e}"
        print(reply)
        elapsed_time = round(time.time() - start_time, 4)
        response_dict={
            'response': reply,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'time_elapsed': elapsed_time,
        }
        return response_dict 


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
    parser.add_argument("--resume", action="store_true", help="Enable if you want to continue from existing results.jsonl file")
    parser.add_argument("--use_verigrad", action="store_true", help="Enable if you want to use verigrad")
    parser.add_argument("--verilogeval", action="store_true", help="Enable if you have the verilogeval dataset")
    parser.add_argument(
        "--prompt_op",
        type=str,
        choices=["Generate", "GenerateCoT", "ReAct", "ReActSimple"],
        default="Generate",
        help="Which LLM prompting technique to use (CoT, Ensemble, etc.)."
    ) # Following the MaAS naming
    parser.add_argument("--refine", action="store_true", help="Enable if you want to refine that op")
    parser.add_argument("--self_refine", action="store_true", help="Enable if you want to use refine directly at runtime")
    parser.add_argument("--ppa_op", action="store_true", help="Enable if you want to use the PPA optimize prompt")
    args = parser.parse_args()

    temperature=args.temperature
    top_p = args.top_p
    max_tokens=args.max_tokens
    resume=args.resume

    model_name = args.model_name
    num_samples = args.num_samples
    batch_size = args.batch_size
    verilogeval = args.verilogeval
    prompt_op = args.prompt_op
    refine = args.refine
    self_refine = args.self_refine
    ppa_op = args.ppa_op
    if ppa_op:
        if prompt_op != "GenerateCoT":
            sys.stderr.write("[ERROR] PPA Prompting is supported as GenerateCoT variant. Relaunch with --prompt_op GenerateCoT\n")
            raise SystemExit(1)
    

    use_verigrad = args.use_verigrad
    use_vllm = args.use_vllm
    openai_reasoning_effort = args.openai_reasoning_effort
    vllm_reasoning = args.vllm_reasoning

    # 0) Pre-flight: fail fast if yosys doesn’t exist
    if self_refine or refine:
        if shutil.which("yosys") is None:
            sys.stderr.write("[ERROR] `yosys` not found on PATH — aborting! Make sure you export it!!\n")
            raise SystemExit(1)

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
                            refine, self_refine, openai_reasoning_effort, ppa_op)

    if resume and os.path.exists(results_file):
        existing_results = load_jsonl_file(results_file)
        already_done = len(existing_results) # // num_samples: will match since in dir-name!
    else:
        already_done = 0
        with open(results_file, "w") as f:
            pass # reset! their code appends indefinitely! OMG!

    # Under that dir, have the tmp yosys files....
    tmpfiles_yosys_path = os.path.join(results_path, "tmp")
    os.makedirs(tmpfiles_yosys_path, exist_ok=True)

    # Source (for refine if set!)
    source_yosys_syntaxchecks_filename = None
    source_yosys_syntaxchecks_results = None
    source_results_file = source_results_path = None
    if refine:
        source_results_file, source_results_path =\
            get_results_filepath(model_name, num_samples, vllm_reasoning, 
                                use_vllm, prompt_op, benchmark_results_dest,
                                False, self_refine, openai_reasoning_effort)
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
    for i in tqdm(range(0, len(question_list), batch_size), total=num_batches, desc="Solving verilog"):
        if i < already_done: continue

        questions_batch = question_list[i : i + batch_size]
        questions_batch_prompts = []
        questions_skip = []
        for j, q in enumerate(questions_batch):

            if prompt_op == "Generate":
                q_prompt = q + GENERATE_PROMPT
            elif prompt_op == "GenerateCoT":
                q_prompt = GENERATE_COT_PROMPT.format(code_task=q) if not ppa_op else GENERATE_COT_PROMPT_PPA.format(code_task=q)
            elif prompt_op == "ReAct":
                q_prompt = REACT_PROMPT.format(code_task=q)
            elif prompt_op == "ReActSimple":
                q_prompt = REACT_PROMPT_SIMPLE.format(code_task=q)
            

            q_skip = False
            if refine:
                idx = i + j
                q_id = idx // num_samples
                sample_id = idx % num_samples
                source_result = get_result_entry(source_results, q_id, sample_id)
                source_yosys_syntaxcheck = get_result_entry(source_yosys_syntaxchecks_results, q_id, sample_id)
                success=source_yosys_syntaxcheck['success']
                error_log=source_yosys_syntaxcheck['error_log']
                if success:
                    q_skip = True
                else:
                    q_prompt = REFINE_PROMPT.format(code_task=q, solution=source_result['generated_code'], test_fail=error_log)

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

        llm_responses_refined = copy.deepcopy(llm_responses)

        if self_refine:

            max_retries = 3
            retry_cnt = 0
            questions_idx_succeeded = []
            questions_succeeded = [False for _ in questions_batch_prompts]

            while retry_cnt<=max_retries and (not all(questions_succeeded)):

                syntax_batch_runs = [
                    yosys_syntax_check(tmpfiles_yosys_path, extract_code_block(llm_response['response']), skip_call=questions_succeeded[j])
                    for j, llm_response in enumerate(llm_responses)
                ]
                yosys_checkresults = loop.run_until_complete(asyncio.gather(*syntax_batch_runs))

                # Go again!
                questions_succeeded = []
                for j, yosys_checkresult in enumerate(yosys_checkresults):
                    # print(j, yosys_checkresult)
                    success = yosys_checkresult['success']
                    questions_succeeded.append(success)
                    if j in questions_idx_succeeded:
                        assert llm_responses[j]['response'] == "Skipped"
                    if success:
                        if j not in questions_idx_succeeded:
                            questions_succeeded[j] = True 
                            llm_responses_refined[j]['response'] = copy.deepcopy(llm_responses[j]['response'])
                            questions_idx_succeeded.append(j)
                        else:
                            assert yosys_checkresult['error_log'] == "Skipped"
                    
                    llm_responses_refined[j]['prompt_tokens'] += llm_responses[j]['prompt_tokens'] # will be 0 if skipped anyway
                    llm_responses_refined[j]['completion_tokens'] += llm_responses[j]['completion_tokens'] # will be 0 if skipped anyway
                    llm_responses_refined[j]['total_tokens'] += llm_responses[j]['total_tokens'] # will be 0 if skipped anyway
                    llm_responses_refined[j]['time_elapsed'] += llm_responses[j]['time_elapsed'] # will be 0 if skipped anyway
                    
                if use_vllm:
                    batch_runs = [
                        get_vllm_response(q, model_name, temperature=temperature, vllm_reasoning=vllm_reasoning, skip_call=questions_succeeded[j]) 
                        for j, q in enumerate(questions_batch_prompts)
                    ]
                else:
                    batch_runs = [
                        get_openai_response(q, model_name, openai_reasoning_effort=openai_reasoning_effort, skip_call=questions_succeeded[j]) 
                        for j, q in enumerate(questions_batch_prompts)
                    ]

                llm_responses = loop.run_until_complete(asyncio.gather(*batch_runs))
                retry_cnt+=1



        for j, llm_response in enumerate(llm_responses_refined):
            idx = i + j
            benchmark_dict = verified_benchmark_dict_list[idx]
            question = question_list[idx]
            q_id = idx // num_samples
            sample_id = idx % num_samples
        
            if refine:
                source_result = get_result_entry(source_results, q_id, sample_id)
                source_yosys_syntaxcheck = get_result_entry(source_yosys_syntaxchecks_results, q_id, sample_id)
                success=source_yosys_syntaxcheck['success']
                error_log=source_yosys_syntaxcheck['error_log']
                if success:
                    assert llm_response['response'] == "Skipped"
                    generated_code = source_result['generated_code'] # reuse!
                    llm_response_final = source_result['full_response'] # reuse!
                    elapsed_time = source_result['time_elapsed'] # reuse!
                    prompt_tokens = source_result['prompt_tokens'] # reuse!
                    completion_tokens = source_result['completion_tokens'] # reuse!
                    total_tokens = source_result['total_tokens'] # reuse!
                else:
                    generated_code = extract_code_block(llm_response['response'])
                    llm_response_final = llm_response['response']
                    elapsed_time = llm_response['time_elapsed']
                    prompt_tokens = llm_response['prompt_tokens']
                    completion_tokens = llm_response['completion_tokens']
                    total_tokens = llm_response['total_tokens']
            else:
                generated_code = extract_code_block(llm_response['response'])
                llm_response_final = llm_response['response']
                elapsed_time = llm_response['time_elapsed']
                prompt_tokens = llm_response['prompt_tokens']
                completion_tokens = llm_response['completion_tokens']
                total_tokens = llm_response['total_tokens']
            
            reply_dict = {
                "q_id": q_id,
                "sample_id": sample_id,
                "time_elapsed": elapsed_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "full_response": llm_response_final,
                "generated_code": generated_code,
                "ground_truth": benchmark_dict['ground_truth']
            }
            with open(results_file, "a") as f:
                f.write(json.dumps(reply_dict) + "\n")

