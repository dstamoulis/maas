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

from verithoughts_utils import extract_code_block, savefile, load_jsonl, rename_modules_and_instantiations, pass_at_k, get_result_entry

from generate_allops import get_results_filepath
import pprint

def debug_pretty_print(question_id, sample_id, result, yosys_checkresult_dict, reflection_result=None):
    """
    Pretty-print debugging information for a single sample.
    """
    pp = pprint.PrettyPrinter(indent=2, width=80)
    
    print(f"{'='*80}")
    print(f"Question ID   : {question_id}")
    print(f"Sample ID     : {sample_id}")
    print(f"{'-'*80}")
    
    print("Generated Code:")
    print(result.get('generated_code', '<none>'))
    print(f"{'-'*80}")
    
    print("Ground Truth:")
    print(result.get('ground_truth', '<none>'))
    print(f"{'-'*80}")
    
    print("Yosys Check  :")
    print(f"  Success    : {yosys_checkresult_dict.get('success')}")
    print("  Error Dict :")
    pp.pprint(yosys_checkresult_dict.get('error_dict', {}))
    print(f"{'-'*80}")
    
    if reflection_result:
        print("Reflection Result:")
        pp.pprint(reflection_result)
        print(f"{'='*80}\n")


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
1. Exactly which part of the generated design failed and why (e.g. syntax error, mismatched logic in module 'adder', etc.).  
2. How you would correct it—describe the specific fix rather than re-posting the entire module.  

Keep your explanation as concise as possible, focusing on root cause and remedy.
"""
    return prompt

def reflect_on_yosysrun(generated_code, ground_truth, error_log):

    reflection_prompt = build_failure_diagnosis_prompt(
        generated_code, ground_truth, error_log
    )    
    return get_gpt_reflection(reflection_prompt)



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

    try:
        result = get_result_entry(results_data, question_id, sample_id)
        yosys_checkresult_dict = get_result_entry(yosys_evals_results, question_id, sample_id)
        # print("Found entry:", entry)
    except ValueError as e:
        print("Error:", e)
        exit()

    reflection_result = None
    if gpt_reflect:
        if not yosys_checkresult_dict['success']:
            reflection_result = \
                reflect_on_yosysrun(result['generated_code'], result['ground_truth'], yosys_checkresult_dict)
        else:
            print("Successful run; no need to GPT-reflect")

    debug_pretty_print(question_id, sample_id, result, yosys_checkresult_dict, reflection_result)