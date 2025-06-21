import asyncio
import threading
import time
import torch
import uuid
import platform
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple, Literal

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

import maas.ext.maas.scripts.optimized.VeriThoughts.train.template.prompt as prompt_custom
from maas.ext.maas.scripts.utils import extract_verilog_code_block, save_verilogfile, clear_verilogfile, rename_modules_and_instantiations
from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.logs import logger
from maas.utils.sanitize import sanitize

# TODO! BUILD THIS!! VeriThoughts doesn't have entry point!!

class VeriThoughtsBenchmark(BaseBenchmark):
    def __init__(self, 
                name: str,
                file_path: str,
                log_path: str,
                batch_size: int,
                controller: torch.nn.Module,
                operator_embeddings: List[List[float]],
                optimizer: torch.optim.Optimizer,):
        super().__init__(name, file_path, log_path, batch_size, controller, operator_embeddings, optimizer)

    class TimeoutError(Exception):
        pass

    async def check_solution(self, solution, canonical_solution, timeout=150):

        # 0) Pre-flight: fail fast if yosys doesn’t exist
        if shutil.which("yosys") is None:
            logger.info("Error loading yosys. Exiting")
            sys.stderr.write("[ERROR] `yosys` not found on PATH — aborting! Make sure you export it!!\n")
            raise SystemExit(1)

        solution = extract_verilog_code_block(solution) # even thu we've extract CODE BEGIN/END already
        canonical_solution = extract_verilog_code_block(canonical_solution) # just in case

        # solution .v as file
        verilog_gen_file = f"logs/verilog_gen/verilog_gen_{uuid.uuid4().hex}.v"
        save_verilogfile(verilog_gen_file, solution)

        # ground-truth .v as file
        verilog_gt_file = f"logs/verilog_truth/verilog_truth_{uuid.uuid4().hex}.v"
        modified_module_golden, mod_module_list = rename_modules_and_instantiations(canonical_solution)
        save_verilogfile(verilog_gt_file, modified_module_golden)

        # yosys script (continuously updated!)
        yosys_equivalence_check_file = f"logs/equivalence_check/equivalence_check_{uuid.uuid4().hex}.ys"

        try:

            yosys_checks_list = []
            yosys_checks_dict = []
            yosys_stdout_dict = []
            yosys_stderr_dict = []
            for original_module_name in mod_module_list:

                module_name = mod_module_list[original_module_name]
                yosys_equivalence_check_script = f"""
                read_verilog {verilog_gt_file}
                read_verilog {verilog_gen_file}
                prep; proc; opt; memory;
                clk2fflogic;
                miter -equiv -flatten {module_name} {original_module_name} miter
                sat -seq 50 -verify -prove trigger 0 -show-all -show-inputs -show-outputs -set-init-zero miter
                """
                save_verilogfile(yosys_equivalence_check_file, yosys_equivalence_check_script)

                proc = await asyncio.create_subprocess_exec(
                    "yosys", "-s", f"{yosys_equivalence_check_file}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
                log_stdout = stdout.decode()
                log_stderr = stderr.decode()
                success = (proc.returncode == 0)
                yosys_checks_list.append(success)
                yosys_checks_dict.append({original_module_name: success})
                yosys_stdout_dict.append({original_module_name: log_stdout})
                if not success and log_stderr:
                    yosys_stderr_dict.append({original_module_name: log_stderr})

        except self.TimeoutError:
            
            logger.info("Timed out while running yosys. Exiting")
            sys.stderr.write(f"[ERROR] Timed out while running `yosys` — aborting\n")
            raise SystemExit(1)

        except Exception as e:
            logger.info(f"Error running yosys. Exiting: {e}")
            sys.stderr.write(f"[ERROR] Error running `yosys` — aborting: {e}\n")
            raise SystemExit(1)

        if all(yosys_checks_list):
            result = (self.PASS, "The solution passed all test cases.")
        else:
            yosys_stderrs = "\n".join(
                        f"{mod}: {err}" 
                        for entry in yosys_stderr_dict 
                        for mod, err in entry.items()
                    )
            error_message = f"For solution: {solution}.\nPer module, yosys returned Errors: {yosys_stderrs}"
            result = (self.FAIL, error_message)

        clear_verilogfile(verilog_gen_file)
        clear_verilogfile(verilog_gt_file)
        clear_verilogfile(yosys_equivalence_check_file)

        return result

    @retry(stop=stop_after_attempt(20), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, prompt):
        # Generate output with a timeout of 200 seconds
        return await asyncio.wait_for(graph(prompt, self.log_path), timeout=1500)

    def get_verithoughts_prompt_solution_pair(self, data: dict):
        if "instruction" in data:
            return data["instruction"], data["output"]
        elif "question" in data:
            return data["question"]+prompt_custom.INSTR_SIMPLE, data["ground_truth"]
        return None, None

    async def evaluate_problem(self, data: dict, graph: Callable):

        input_text, canonical_solution = self.get_verithoughts_prompt_solution_pair(data)
        if input_text is None:
            logger.info("Error loading query-solution from .jsonl. Skipping this sample.")
            return "", "Timeout", "", 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device)

        expected_output = (
            "\nCorrect Solution:"
            + "\n\n"
            + canonical_solution
        )

        try:

            # print(input_text) #, canonical_solution)
            # prediction = canonical_solution
            prediction, cost, logprob = await self._generate_output(graph, input_text)

            # print(prediction)
            # print(cost)
            # print(logprob)
            # exit()
            # return "input_text", "Timeout", "expected_output", 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device)
            
            if not prediction:
                raise ValueError("Prediction is empty")

            # Match solution correctness using yosys; follows VeriThoughts scripting!
            ret = await self.check_solution(prediction, canonical_solution)

            if not isinstance(ret, (list, tuple)) or len(ret) < 2:
                logger.info("Invalid return value from check_solution.")
            test_case_details = ret[1]
            expected_output = test_case_details + expected_output
            score = 1.0 if ret[0] == self.PASS else 0.0

            # print(ret)
            # print("CMON")
            # print(score)
            # exit()

            if score == 0:
                self.log_mismatch(input_text, expected_output, prediction, score)

            return input_text, prediction, expected_output, score, cost, logprob

        except asyncio.TimeoutError:
            logger.info("Timeout error. Skipping this sample.")
            return input_text, "Timeout", expected_output, 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device)

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}") 
            return input_text, "Timeout", expected_output, 0.0, 0.0, torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        return 0.0, prediction

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost", "logprob"]

