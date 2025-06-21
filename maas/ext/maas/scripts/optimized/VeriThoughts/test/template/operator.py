import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple

import asyncio
import tempfile
import os
import shutil
import uuid

from maas.ext.maas.scripts.optimized.VeriThoughts.train.template.operator_an import GenerateOp, ScEnsembleOp, ReflectionTestOp, SelfRefineOp
from maas.ext.maas.scripts.optimized.VeriThoughts.train.template.op_prompt import REFLECTION_ON_PUBLIC_TEST_PROMPT, SC_ENSEMBLE_PROMPT, SELFREFINE_PROMPT
from maas.ext.maas.scripts.utils import extract_verilog_code_block, save_verilogfile, clear_verilogfile
from maas.actions.action_node import ActionNode
from maas.llm import LLM
from maas.logs import logger
import re


class Operator:
    def __init__(self, llm: LLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        fill_kwargs = {"context": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)
        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        return node.instruct_content.model_dump()

class CustomCodeGenerate(Operator):
    def __init__(self, llm: LLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="verilog_fill")
        return response

class Generate(Operator):
    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(llm, name)

    async def __call__(self, problem, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="verilog_fill")
        return response
    
class GenerateCoT(Operator):
    def __init__(self, llm: LLM, name: str = "GenerateCoT"):
        super().__init__(llm, name)

    async def __call__(self, problem, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="verilog_fill")
        return response

class MultiGenerateCoT(Operator):
    def __init__(self, llm: LLM, name: str = "MultiGenerateCoT"):
        super().__init__(llm, name)

    async def __call__(self, problem, instruction):
        prompt = instruction + problem
        
        response1 = await self._fill_node(GenerateOp, prompt, mode="verilog_fill")
        response2 = await self._fill_node(GenerateOp, prompt, mode="verilog_fill")
        response3 = await self._fill_node(GenerateOp, prompt, mode="verilog_fill")
        
        return {"response": [response1, response2, response3]}
    
class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """
    def __init__(self, llm: LLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}


class Test(Operator):
    def __init__(self, llm: LLM, name: str = "Test"):
        super().__init__(llm, name)

    async def exec_code_with_yosys(self, verilog_path: str, timeout: int = 150) -> (bool, str, str):

        """
        Run Yosys on `verilog_path`, non-blockingly, and return (success, log_output).
        """
        # 0) Pre-flight: fail fast if yosys doesn’t exist
        if shutil.which("yosys") is None:
            logger.info("Error loading yosys. Exiting")
            sys.stderr.write("[ERROR] `yosys` not found on PATH — aborting! Make sure you export it!!\n")
            raise SystemExit(1)

        # build the inline yosys script
        yosys_script = (
            f"read_verilog {verilog_path}; "
            "hierarchy -check; "
            "proc; "
            "opt; "
            "stat"
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                "yosys", "-p", yosys_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as e:
            logger.info("Error running yosys. Exiting")
            sys.stderr.write(f"[ERROR] Error running `yosys` — aborting: {e}\n")
            raise SystemExit(1)

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        except Exception as e:
            logger.info("Error running yosys. Exiting")
            sys.stderr.write(f"[ERROR] Error running `yosys` — aborting: {e}\n")
            raise SystemExit(1)
        except asyncio.TimeoutError:
            logger.info("Timed out while running yosys. Exiting")
            sys.stderr.write(f"[ERROR] Error running `yosys` — timed out after {timeout}s\n")
            raise SystemExit(1)

        log_stdout = stdout.decode()
        log_stderr = stderr.decode()
        success = (proc.returncode == 0)

        return success, log_stdout, log_stderr


    async def __call__(
        self, problem, solution, test_loop: int = 1
    ):
        """
        "Test": {
        "description": "Test the solution with yosys, if the solution is correct, return 'no error', if the solution is incorrect, return reflect on the solution and the error information",
        "interface": "test(problem: str, solution: str) -> str"
        }
        """
        temp_verilog_path = f"logs/verilog_tester/verilog_tester_{uuid.uuid4().hex}.v"

        for _ in range(test_loop):

            verilog_code = extract_verilog_code_block(solution) # even thu we've extract CODE BEGIN/END already
            save_verilogfile(temp_verilog_path, verilog_code)
    
            # Run the yosys check!
            success, yosys_log_stdout, yosys_log_stderr = await self.exec_code_with_yosys(temp_verilog_path)
            if success:
                clear_verilogfile(temp_verilog_path)
                return {"result": True, "solution": solution}
            elif test_loop == 1:
                clear_verilogfile(temp_verilog_path)
                return {"result": False, "solution": solution}
            else:
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=yosys_log_stdout,
                    test_fail=yosys_log_stderr,
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="verilog_fill")
                solution = response["reflection_and_solution"]
        
        success, yosys_log_stdout, yosys_log_stderr = await self.exec_code_with_yosys(temp_verilog_path)
        clear_verilogfile(temp_verilog_path)
        if success:
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}
        
class SelfRefine(Operator):
    def __init__(self, llm: LLM, name: str = "SelfRefine"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution):
        prompt = SELFREFINE_PROMPT.format(problem=problem, solution=solution)
        response = await self._fill_node(SelfRefineOp, prompt, mode="verilog_fill")
        return response
    
class EarlyStop(Operator):
    def __init__(self, llm: LLM, name: str = "EarlyStop"):
        super().__init__(llm, name)

    async def __call__(self):
        return NotImplementedError
