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


def prep_output_dirtree(tmpfiles_genus_path):

    os.makedirs(tmpfiles_genus_path, exist_ok=True)

    area_dir = os.path.join(tmpfiles_genus_path, "area_analysis")
    gate_dir = os.path.join(tmpfiles_genus_path, "gate_analysis")
    delay_dir = os.path.join(tmpfiles_genus_path, "delay_analysis")
    power_dir = os.path.join(tmpfiles_genus_path, "power_analysis")
    verilog_dir = os.path.join(tmpfiles_genus_path, "synth")
    log_dir =  os.path.join(tmpfiles_genus_path, "log")

    for directory in [area_dir, gate_dir, delay_dir, power_dir, verilog_dir, log_dir]:
        os.makedirs(directory, exist_ok=True)


async def genus_synth(generated_code, liberty, tmpfiles_genus_path, keep_log_stdout=False):

    # generated .v as file
    tmp_fileid = uuid.uuid4().hex
    verilog_gen_file = os.path.join(tmpfiles_genus_path, f"verilog_synth_{tmp_fileid}.v")
    savefile(verilog_gen_file, generated_code)

    metrics = {
        'area': -1, 
        'gate_count': -1,
        'comb_cells': -1,
        'seq_cells': -1,
        'delay': -1, 
        'static_power': -1, 
        'switching_power': -1, 
        'time': -1
    }

    area_dir = os.path.join(tmpfiles_genus_path, "area_analysis")
    gate_dir = os.path.join(tmpfiles_genus_path, "gate_analysis")
    delay_dir = os.path.join(tmpfiles_genus_path, "delay_analysis")
    power_dir = os.path.join(tmpfiles_genus_path, "power_analysis")
    verilog_dir = os.path.join(tmpfiles_genus_path, "synth")
    log_dir =  os.path.join(tmpfiles_genus_path, "log")

    has_clk, clk_port = has_clk_signal(verilog_gen_file)

    file_name = os.path.basename(verilog_gen_file)
    output_verilog_synth = os.path.join(verilog_dir, file_name)
    area_report = os.path.join(area_dir, f"{file_name}.txt")
    gate_report = os.path.join(gate_dir, f"{file_name}.txt")
    timing_report = os.path.join(delay_dir, f"{file_name}.txt")
    power_report = os.path.join(power_dir, f"{file_name}.txt")

    if has_clk: 
        sdc_timing = f"""
            create_clock -name {clk_name} -period 0 [get_ports {clk_name}]; 
            set_input_delay -clock {clk_name} 0 [remove_from_collection [all_inputs] [get_ports {clk_name}]]; 
            set_output_delay -clock {clk_name} 0 [all_outputs];
        """
    else:
        sdc_timing = f"""
            create_clock -name clk -period 0;
            set_input_delay -clock clk 0 [all_inputs]; 
            set_output_delay -clock clk 0 [all_outputs]; 
        """        

    # savefile(sdc_file, synth_sdc_script)
    genus_script = f'''
        read_libs {liberty};
        read_hdl -sv {verilog_gen_file};
        elaborate;
        flatten;
        synthesize -to_generic -effort high;
        syn_map;
        syn_opt; 
        report_area > {area_report};
        report_gates > {gate_report};
        write_hdl -mapped > {output_verilog_synth};
        {sdc_timing}
        report_timing > {timing_report};
        report_power -by_libcell > {power_report};
        exit;
    '''

    start = time.time()

    full_command = [
        'genus', 
        '-abort_on_error', 
        '-execute', 
        genus_script, 
        '-log', 
        '/dev/null'
    ]
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
                timeout=120*2
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
        log_stderr = f"Genus failed: {e}"
        success = False

    end = time.time()
    metrics['time'] = end-start 
    
    if os.path.isfile(area_report):
        cells, metrics['area']= parse_area(area_report)
        metrics['gate_count']= sum(cell['count'] for cell in cells.values())
        metrics['comb_cells'] = sum(cell['count'] for name, cell in cells.items() if 'DF' not in name)
        metrics['seq_cells'] = metrics['gate_count'] - metrics['comb_cells']
        
    genus_checkresult_dict = {}
    genus_checkresult_dict['genus_success'] = success
    genus_checkresult_dict['sta_success'] = "NA"

    genus_checkresult_dict['area'] = metrics['area']
    genus_checkresult_dict['delay'] = metrics['delay']
    genus_checkresult_dict['static_power'] = metrics['static_power']
    genus_checkresult_dict['switching_power'] = metrics['switching_power']

    genus_checkresult_dict['genus_return_code'] = log_returncode
    genus_checkresult_dict['genus_error_log'] = log_stderr

    if os.path.isfile(timing_report):
        metrics['delay']=parse_delay(timing_report) 
    
    if os.path.isfile(power_report):
        power=parse_power(power_report)
        metrics['static_power'] = power['leakage']
        metrics['switching_power'] = power['switching']

    genus_checkresult_dict['delay'] = metrics['delay']
    genus_checkresult_dict['static_power'] = metrics['static_power']
    genus_checkresult_dict['switching_power'] = metrics['switching_power']

    clear_verilogfile(verilog_gen_file)
    clear_verilogfile(output_verilog_synth)
    clear_verilogfile(area_report)
    clear_verilogfile(gate_report)
    clear_verilogfile(timing_report)
    clear_verilogfile(power_report)

    return genus_checkresult_dict


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

    if shutil.which("genus") is None:
        sys.stderr.write("[ERROR] `genus` not found — aborting! Make sure you have it?!?!\n")
        raise SystemExit(1)

    # NO! benchmark_data = load_json(args.benchmark_path)
    # NO! parser.add_argument("--benchmark_path", type=str, default="VeriThoughtsBenchmark", help="Path to the benchmark jsonl")
    # YES! Login using e.g. `huggingface-cli login` to access this dataset
    if verilogeval:
        # benchmark_data = load_dataset("dakies/nvlabs-verilogeval-v2-spec-to-rtl", split="test")
        benchmark_results_dest = "benchmark_results_verilogeval"
    else:
        # benchmark_data = load_dataset("wilyub/VeriThoughtsBenchmark", split="train")
        benchmark_results_dest = "benchmark_results"

    # Directory: benchmark_results/{model_name}/
    results_file, results_path = \
        get_results_filepath(model_name, num_samples, vllm_reasoning, 
                            use_vllm, prompt_op, benchmark_results_dest,
                            refine, self_refine, openai_reasoning_effort, ppa_op)
    results_data = load_jsonl(results_file)
    # Under that dir, have the tmp yosys files....
    tmpfiles_genus_path = os.path.join(results_path, "tmp_synth")
    prep_output_dirtree(tmpfiles_genus_path)
    # Same for yosys results ... c'mon....
    genus_synth_results_filename = os.path.join(results_path, "genus_ppa.jsonl")
    with open(genus_synth_results_filename, "w") as f:
        pass # reset! their code appends indefinitely! OMG!

    loop = asyncio.get_event_loop()
    num_batches = (len(results_data) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(results_data), batch_size), total=num_batches, desc="Running yosys checks batches!!"):

        results_batch = results_data[i : i + batch_size]
        batch_runs = [
            genus_synth(
                generated_code=result['generated_code'], 
                liberty=liberty,
                tmpfiles_genus_path=tmpfiles_genus_path
            ) 
            for result in results_batch
        ]
        genus_synth_results = loop.run_until_complete(asyncio.gather(*batch_runs))

        for j, genus_synth_result in enumerate(genus_synth_results):
            idx = i + j
            q_id = idx // num_samples
            sample_id = idx % num_samples
            genus_synth_result_dict = {
                "q_id": q_id,
                "sample_id": sample_id,
                "genus_success": genus_synth_result["genus_success"],
                "sta_success": genus_synth_result["sta_success"],
                "area": genus_synth_result["area"],
                "delay": genus_synth_result["delay"],
                "static_power": genus_synth_result["static_power"],
                "switching_power": genus_synth_result["switching_power"],
                "genus_return_code": genus_synth_result["genus_return_code"],
                "genus_error_log": genus_synth_result["genus_error_log"],
                "sta_return_code": genus_synth_result["sta_return_code"],
                "sta_error_log": genus_synth_result["sta_error_log"],
            }

            with open(genus_synth_results_filename, "a") as f:
                f.write(json.dumps(genus_synth_result_dict) + "\n")
