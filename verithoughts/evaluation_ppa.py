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

from verithoughts_utils import extract_code_block, savefile, load_jsonl, rename_modules_and_instantiations, pass_at_k, clear_verilogfile

from yosys_utils import has_clk_signal, get_top_module
from yosys_utils import parse_power, parse_delay, get_delay, parse_area


SDC_CONSTRAINTS_CLOCK= """
    create_clock -name {clock_name} -period {clock_period} [get_ports {clock_port}]; 
    set clk_input [get_port {clock_port}]
    set clk_indx [lsearch [all_inputs] $clk_input]
    set all_inputs_wo_clk [lreplace [all_inputs] $clk_indx $clk_indx ""]
    set_input_delay -clock {clock_name} 0 $all_inputs_wo_clk
    set_output_delay -clock {clock_name} 0 [all_outputs]
    set_max_fanout {max_fanout} [current_design]
"""

SDC_CONSTRAINTS_WO_CLOCK = """
create_clock -name {clock_name} -period {clock_period};
set_input_delay -clock {clock_name} 0 [all_inputs]; 
set_output_delay -clock {clock_name} 0 [all_outputs]; 
set_max_fanout {max_fanout} [current_design];
"""

SDC_DRIVER = """
set_driving_cell -lib_cell {driving_cell} -pin {driving_cell_pin} $all_inputs_wo_clk
set_driving_cell -lib_cell {driving_cell} -pin {driving_cell_pin} $clk_input
"""


def prep_output_dirtree(tmpfiles_yosys_path):

    os.makedirs(tmpfiles_yosys_path, exist_ok=True)

    area_dir = os.path.join(tmpfiles_yosys_path, "area_analysis")
    gate_dir = os.path.join(tmpfiles_yosys_path, "gate_analysis")
    delay_dir = os.path.join(tmpfiles_yosys_path, "delay_analysis")
    power_dir = os.path.join(tmpfiles_yosys_path, "power_analysis")
    verilog_dir = os.path.join(tmpfiles_yosys_path, "synth")
    sdc_dir = os.path.join(tmpfiles_yosys_path, "sdc")
    log_dir =  os.path.join(tmpfiles_yosys_path, "log")

    for directory in [area_dir, gate_dir, delay_dir, power_dir, verilog_dir, sdc_dir, log_dir]:
        os.makedirs(directory, exist_ok=True)


async def yosys_synth(generated_code, liberty, target_clock_period, tmpfiles_yosys_path, keep_log_stdout=False):

    # generated .v as file
    tmp_fileid = uuid.uuid4().hex
    verilog_gen_file = os.path.join(tmpfiles_yosys_path, f"verilog_synth_{tmp_fileid}.v")
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

    area_dir = os.path.join(tmpfiles_yosys_path, "area_analysis")
    gate_dir = os.path.join(tmpfiles_yosys_path, "gate_analysis")
    delay_dir = os.path.join(tmpfiles_yosys_path, "delay_analysis")
    power_dir = os.path.join(tmpfiles_yosys_path, "power_analysis")
    verilog_dir = os.path.join(tmpfiles_yosys_path, "synth")
    sdc_dir = os.path.join(tmpfiles_yosys_path, "sdc")
    log_dir =  os.path.join(tmpfiles_yosys_path, "log")


    has_clk, clk_port = has_clk_signal(verilog_gen_file)

    file_name = os.path.basename(verilog_gen_file)
    output_verilog_synth = os.path.join(verilog_dir, file_name)
    sdc_file = os.path.join(sdc_dir, f"{file_name[:-2]}.sdc")
    area_report = os.path.join(area_dir, f"{file_name}.txt")
    timing_report = os.path.join(delay_dir, f"{file_name}.txt")
    power_report = os.path.join(power_dir, f"{file_name}.txt")

    tri_buf_map = f"yosys_synth/tribuff_map.v"
    latch_map = f"yosys_synth/latch_map.v"

    if has_clk: 
        synth_sdc_script = SDC_CONSTRAINTS_CLOCK.format(
            clock_name="clk", 
            clock_port=clk_port, 
            clock_period=target_clock_period, 
            max_fanout=10, 
        )
        delay_sdc = SDC_CONSTRAINTS_CLOCK.format(
            clock_name="clk",
            clock_port=clk_port,
            clock_period=0,
            max_fanout=10
        )
        power_sdc = SDC_CONSTRAINTS_CLOCK.format(
            has_clk, 
            clock_name="myClock", 
            clock_port=clk_port, 
            clock_period=1, 
            max_fanout=10, 
            driving_cell="sky130_fd_sc_hd__inv_2", 
            driving_cell_pin="Y"
        )
    else:
        synth_sdc_script = SDC_CONSTRAINTS_WO_CLOCK.format(
            clock_name="clk", 
            clock_period=target_clock_period, 
            max_fanout=10, 
        )
        delay_sdc = SDC_CONSTRAINTS_WO_CLOCK.format(
            clock_name="clk",
            clock_period=0,
            max_fanout=10
        )
        power_sdc = SDC_CONSTRAINTS_WO_CLOCK.format(
            has_clk, 
            clock_name="myClock", 
            clock_period=1, 
            max_fanout=10, 
        )
        
    synth_sdc_script += SDC_DRIVER.format(
        driving_cell="sky130_fd_sc_hd__inv_2",
        driving_cell_pin="Y"
    )
    
    delay_sdc += SDC_DRIVER.format(
        driving_cell="sky130_fd_sc_hd__inv_2",
        driving_cell_pin="Y"
    )
    
    power_sdc += SDC_DRIVER.format(
        driving_cell="sky130_fd_sc_hd__inv_2",
        driving_cell_pin="Y"
    )
        
    savefile(sdc_file, synth_sdc_script)

    yosys_script = f"""
        # Read the liberty file
        read_liberty -lib -ignore_miss_dir -setattr blackbox {liberty}
        
        # Read the design
        read_verilog -sv {verilog_gen_file}
                
        # Attempt to automatically determine the top module
        hierarchy -check -auto-top

        # Convert the design to a generic netlist
        synth -auto-top -flatten
        
        # map tri-state buffers
        techmap -map {tri_buf_map}

        # map latches
        techmap -map {latch_map}

        # mapping flip-flops
        dfflibmap -liberty {liberty}

        # abc optimizations
        abc -liberty {liberty}

        # Technology mapping of constant hi- and/or lo-drivers
        hilomap -singleton \
        -hicell sky130_fd_sc_hd__conb_1 HI \
        -locell sky130_fd_sc_hd__conb_1 LO

        # write synthesized netlist
        opt_clean -purge
       
        # report area 
        tee -o {area_report} stat -liberty {liberty} 
        
        write_verilog -noattr -noexpr -nohex -nodec -defparam  {output_verilog_synth}
    """

    start = time.time()

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
        log_stderr = f"Yosys failed: {e}"
        success = False

    end = time.time()
    metrics['time'] = end-start 
    
    if os.path.isfile(area_report):
        cells, metrics['area']= parse_area(area_report)
        metrics['gate_count']= sum(cell['count'] for cell in cells.values())
        metrics['comb_cells'] = sum(cell['count'] for name, cell in cells.items() if 'df' not in name)
        metrics['seq_cells'] = metrics['gate_count'] - metrics['comb_cells']
        
    yosys_checkresult_dict = {}
    yosys_checkresult_dict['yosys_success'] = success
    yosys_checkresult_dict['sta_success'] = "NA"

    yosys_checkresult_dict['area'] = metrics['area']
    yosys_checkresult_dict['delay'] = metrics['delay']
    yosys_checkresult_dict['static_power'] = metrics['static_power']
    yosys_checkresult_dict['switching_power'] = metrics['switching_power']

    yosys_checkresult_dict['yosys_return_code'] = log_returncode
    yosys_checkresult_dict['yosys_error_log'] = log_stderr

    yosys_checkresult_dict['sta_return_code'] = "NA"
    yosys_checkresult_dict['sta_error_log'] = "NA"

    # report timing and power
    top_module = None 
    if os.path.isfile(output_verilog_synth): 
        top_module = get_top_module(output_verilog_synth)
    
    if top_module is None or not yosys_checkresult_dict['yosys_success']:
        clear_verilogfile(verilog_gen_file)
        clear_verilogfile(output_verilog_synth)
        clear_verilogfile(sdc_file)
        clear_verilogfile(area_report)
        clear_verilogfile(timing_report)
        clear_verilogfile(power_report)
        return yosys_checkresult_dict


    # else run power-timing analysis!
    opensta_script = f"""
        read_liberty {liberty}
        read_verilog {output_verilog_synth}
        link_design {top_module}
        {delay_sdc};
        report_checks -sort_by_slack -path_delay max -fields {{slew cap input nets fanout}} -format full_clock_expanded -group_count 1 > {timing_report}
        {power_sdc}
        set_power_activity -global -activity 0.000000001
        report_power > {power_report}
    """

    full_command = ["sta", "-no_init", "-exit"] #, f"{opensta_script}"]
    log_stdout = log_stderr = ""
    success = log_returncode = None

    # Offload to thread executor
    try:
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                full_command,
                input=opensta_script,
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
        # print(result)
        # print(result.returncode)
        # print(result.stdout)
    except subprocess.TimeoutExpired as e:
        log_returncode = -1
        log_stderr = f"TimeoutExpired: {e}"
        success = False
    except Exception as e:
        log_returncode = -1
        log_stderr = f"STA failed: {e}"
        success = False

    yosys_checkresult_dict['sta_success'] = success
    yosys_checkresult_dict['sta_return_code'] = log_returncode
    yosys_checkresult_dict['sta_error_log'] = log_stderr

    if os.path.isfile(timing_report):
        metrics['delay']=get_delay(timing_report) 
    
    if os.path.isfile(power_report):
        power=parse_power(power_report)
        metrics['static_power'] = power['static']
        metrics['switching_power'] = power['switching']

    yosys_checkresult_dict['delay'] = metrics['delay']
    yosys_checkresult_dict['static_power'] = metrics['static_power']
    yosys_checkresult_dict['switching_power'] = metrics['switching_power']

    clear_verilogfile(verilog_gen_file)
    clear_verilogfile(output_verilog_synth)
    clear_verilogfile(sdc_file)
    clear_verilogfile(area_report)
    clear_verilogfile(timing_report)
    clear_verilogfile(power_report)

    return yosys_checkresult_dict



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arg Parse")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="HF model name")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per question")
    parser.add_argument("--batch_size", type=int, default=20, help="Number of yosys runs to run concurrently")
    parser.add_argument("--enable_reasoning", action="store_true", help="Enable if you have a reasoning mode triggered by <think>")
    parser.add_argument(
        "--prompt_op",
        type=str,
        choices=["Generate", "GenerateCoT", "MultiGenerateCoT", "ScEnsemble", "Test", "SelfRefine", "EarlyStop",  "ReAct"],
        default="Generate",
        help="Which LLM prompting technique to use (CoT, Ensemble, etc.)."
    ) # Following the MaAS naming
    parser.add_argument('--liberty', type=str, default="skywater-pdk/libraries/sky130_fd_sc_hd/latest/timing/sky130_fd_sc_hd__tt_025C_1v80.lib", help="Liberty file to use for synthesis")
    parser.add_argument('--target_clock_period', type=int, help="Target clock period in ns", default=20)
    args = parser.parse_args()

    model_name = args.model_name
    num_samples = args.num_samples
    enable_reasoning = args.enable_reasoning
    batch_size = args.batch_size
    prompt_op = args.prompt_op

    target_clock_period = args.target_clock_period
    liberty = args.liberty

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
    _names_list = [model_name, f"samples_{num_samples}"]
    if enable_reasoning: _names_list.append("reasoning")
    # if use_verigrad: _names_list.append("verigrad")
    if prompt_op != "Generate":
        _names_list.append(prompt_op)
    sub_folder = "-".join(_names_list)
    results_path = os.path.join("benchmark_results", sub_folder)
    os.makedirs(results_path, exist_ok=True)
    results_file = os.path.join(results_path, "results.jsonl")
    results_data = load_jsonl(results_file)
    # Under that dir, have the tmp yosys files....
    tmpfiles_yosys_path = os.path.join(results_path, "tmp_synth")
    prep_output_dirtree(tmpfiles_yosys_path)
    # Same for yosys results ... c'mon....
    yosys_synth_results_filename = os.path.join(results_path, "yosys_ppa.jsonl")
    with open(yosys_synth_results_filename, "w") as f:
        pass # reset! their code appends indefinitely! OMG!

    loop = asyncio.get_event_loop()
    num_batches = (len(results_data) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(results_data), batch_size), total=num_batches, desc="Running yosys checks batches!!"):

        results_batch = results_data[i : i + batch_size]
        batch_runs = [
            yosys_synth(
                generated_code=result['generated_code'], 
                liberty=liberty, 
                target_clock_period=target_clock_period, 
                tmpfiles_yosys_path=tmpfiles_yosys_path
            ) 
            for result in results_batch
        ]
        yosys_synth_results = loop.run_until_complete(asyncio.gather(*batch_runs))

        for j, yosys_synth_result in enumerate(yosys_synth_results):
            idx = i + j
            q_id = idx // num_samples
            sample_id = idx % num_samples
            yosys_synth_result_dict = {
                "q_id": q_id,
                "sample_id": sample_id,
                "yosys_success": yosys_synth_result["yosys_success"],
                "sta_success": yosys_synth_result["sta_success"],
                "area": yosys_synth_result["area"],
                "delay": yosys_synth_result["delay"],
                "static_power": yosys_synth_result["static_power"],
                "switching_power": yosys_synth_result["switching_power"],
                "yosys_return_code": yosys_synth_result["yosys_return_code"],
                "yosys_error_log": yosys_synth_result["yosys_error_log"],
                "sta_return_code": yosys_synth_result["sta_return_code"],
                "sta_error_log": yosys_synth_result["sta_error_log"],
            }

            with open(yosys_synth_results_filename, "a") as f:
                f.write(json.dumps(yosys_synth_result_dict) + "\n")

        # exit()

    # yosys_results_dict = load_jsonl(yosys_synth_results_filename) # Oh my!!!
    # correct_counts = []
    # for i in range(0, len(yosys_results_dict), num_samples):
    #     per_question_yosys_results_dict = yosys_results_dict[i:i+num_samples]
    #     correct_counter = sum(1 for sample in per_question_yosys_results_dict if sample['success']) # One-liner FTW!
    #     correct_counts.append(correct_counter)

    # print("pass@1:", pass_at_k(correct_counts, num_samples, 1))
    # print("pass@5:", pass_at_k(correct_counts, num_samples, 5))
    # print("pass@10:", pass_at_k(correct_counts, num_samples, 10))
