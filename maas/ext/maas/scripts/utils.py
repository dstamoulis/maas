import json
import re
import os
import ast
import random
from enum import Enum
from typing import Any, List, Tuple
from collections import Counter

class CodeDataset(Enum):
    HUMAN_EVAL = "HumanEval"
    MBPP = "MBPP"
    VERITHOUGHTS = "VeriThoughts"

def extract_random_prompt(log_path: str):
    parent_dir = os.path.dirname(log_path)

    prompt_file_path = os.path.join(parent_dir, "template", "op_prompt.py")
    
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt does not exist: {prompt_file_path}")
    
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    
    tree = ast.parse(file_content)
    prompt_dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith('_PROMPT'):
                    if isinstance(node.value, ast.Constant):
                        prompt_content = node.value.value
                    elif isinstance(node.value, ast.Str):
                        prompt_content = node.value.s
                    else:
                        continue
                    prompt_dict[target.id] = prompt_content

    if not prompt_dict:
        return None, None

    prompt_name, prompt_content = random.choice(list(prompt_dict.items()))
    return prompt_name, prompt_content

def update_prompt_in_file(log_path: str, prompt_name: str, prompt_content: str):
    parent_dir = os.path.dirname(log_path)
    prompt_file_path = os.path.join(parent_dir, "template", "op_prompt.py")
    
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt does not exist: {prompt_file_path}")
    
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = rf'{re.escape(prompt_name)}\s*=\s*"""(.*?)"""'
    match = re.search(pattern, content, flags=re.DOTALL)

    if match:
        old_prompt = match.group(1)
        old_placeholders = re.findall(r'{([^{}]+)}', old_prompt)
        new_placeholders = re.findall(r'{([^{}]+)}', prompt_content)
        if Counter(old_placeholders) != Counter(new_placeholders):
            return
    else:
        pass

    new_assignment = f'{prompt_name} = """\n{prompt_content}\n"""'
    
    if re.search(pattern, content, flags=re.DOTALL):
        new_content = re.sub(pattern, new_assignment, content, flags=re.DOTALL)
    else:
        new_content = content.rstrip() + "\n\n" + new_assignment + "\n"
    
    with open(prompt_file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

def extract_test_cases_from_jsonl(entry_point: str, dataset: CodeDataset = CodeDataset.HUMAN_EVAL):
    if dataset == CodeDataset.HUMAN_EVAL.value:
        file_path = "maas/ext/maas/data/humaneval_public_test.jsonl"
        # Retain the original hardcoded test cases
        hardcoded_cases = {
            "find_zero": "",
            "decode_cyclic": "",
            "decode_shift": "",
            "by_length": "",
            "add": "",
            "triangle_area": "",
            "correct_bracketing": "",
            "solve": "",
            "sum_squares": "",
            "starts_one_ends": "",
        }
    elif dataset == CodeDataset.MBPP.value:
        file_path = "maas/ext/maas/data/mbpp_public_test.jsonl"
        hardcoded_cases = {
            "remove_odd": "",
            "replace_spaces": "",
            "snake_to_camel": "",
            "Split": "",
            "swap_List": "",
            "square_Sum": "",
            "sort_sublists": "",
            "unique_sublists": "",
        }
    elif dataset == CodeDataset.VERITHOUGHTS.value:
        file_path = "maas/ext/maas/data/verithoughts_public_test.jsonl"
        hardcoded_cases = {} # TODO: VeriThoughts doesn't have that!!

    # Check if there are hardcoded test cases
    if entry_point in hardcoded_cases:
        return hardcoded_cases[entry_point]

    # If there are no hardcoded test cases, read from the file
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data.get("entry_point") == entry_point:
                return data.get("test")

    return None


def extract_test_cases(docstring: str) -> List[Tuple[str, List[Any], Any]]:
    # Use regular expressions to match test cases, now capturing function names and any output
    pattern = r">>> (\w+)\((.*?)\)\n\s*(.*?)(?=\n|$)"
    matches = re.findall(pattern, docstring, re.DOTALL)

    test_cases = []
    for match in matches:
        func_name, input_str, expected_output = match

        # Process input
        input_list = []
        for item in input_str.split(","):
            item = item.strip()
            try:
                # Try to convert input to numeric type
                if "." in item:
                    input_list.append(float(item))
                else:
                    input_list.append(int(item))
            except ValueError:
                # If unable to convert to numeric, keep as string
                input_list.append(item.strip("'\""))

        # Process output
        try:
            # Try to convert output to numeric or boolean value
            if expected_output.lower() == "true":
                expected_output = True
            elif expected_output.lower() == "false":
                expected_output = False
            elif "." in expected_output:
                expected_output = float(expected_output)
            else:
                expected_output = int(expected_output)
        except ValueError:
            # If unable to convert, keep as string
            expected_output = expected_output.strip("'\"")

        test_cases.append([func_name, input_list, expected_output])

    return test_cases


def test_cases_2_test_functions(solution: str, test_cases: str):
    tester_function = f"""
{solution}

{test_cases}
"""
    return tester_function


def test_case_2_test_function(solution: str, test_case: str, entry_point: str):
    tester_function = f"""
{solution}


def check(candidate):
    {test_case}

def test_check():
    check({entry_point})

test_check()
"""
    return tester_function


def extract_verilog_code_block_depr(text): # THEIR CODE: BUGGY! FAILS to catch ```verilog``` fences!!
    if not text:
        return ""
    pattern = r"CODE BEGIN(.*?)CODE END"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        s = matches[-1]  # take the last match
    else:
        s = text # Return raw (since might be already extracted!)
    return s


# this adapts our ActionNode MaAS sanitizer
# maas.actions.action_node: def verilog_fill
def extract_verilog_code_block(content: str) -> str:
    """
    Extract the last ```verilog``` fenced block if present,
    and/or catch the last CODE BEGIN … CODE END block (which could be present with or without fences),
    or finally everything if neither is found.
    """
    if not content:
        return ""
    code = content.strip()
    # 1) If there are any ```verilog … ``` fences, grab the *last* inner chunk
    fences = re.findall(r"```(?:verilog)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if fences:
        code = fences[-1].strip()

    # # 2) Now, within that (or within the raw content if no fences), look for CODE BEGIN/END
    # blocks = re.findall(r"CODE BEGIN(.*?)CODE END", code, re.DOTALL | re.IGNORECASE)
    # if blocks:
    #     code = blocks[-1].strip()

    # 2) Then last CODE BEGIN…CODE END wins (with or without “// ”)
    blocks = re.findall(
        r"(?://\s*)?CODE\s+BEGIN(.*?)(?://\s*)?CODE\s+END",
        code, re.IGNORECASE | re.DOTALL
    )
    if blocks:
        code = blocks[-1].strip()

    return code


def save_verilogfile(filename, content, fmode='w'):
    with open(filename, fmode) as f:
        f.write(content)

def clear_verilogfile(filename):
    if os.path.exists(filename): 
        os.remove(filename)

def rename_modules_and_instantiations(verilog_code):
    # Step 1: Find all module names (including those with parameters using #(...))
    module_pattern = re.compile(r'\bmodule\s+(\w+)\s*(?:#\s*\(.*?\))?\s*\(', re.DOTALL)
    module_names = module_pattern.findall(verilog_code)

    # Step 2: Create a mapping from old to new names
    rename_map = {name: name + '1' for name in module_names}

    # Step 3: Replace module declarations
    def replace_module_decl(match):
        original_name = match.group(1)
        before = match.group(0)
        return before.replace(original_name, rename_map[original_name], 1)

    verilog_code = module_pattern.sub(replace_module_decl, verilog_code)

    # Step 4: Replace module instantiations (word boundaries)
    for old_name, new_name in rename_map.items():
        instantiation_pattern = re.compile(rf'\b{old_name}\b')
        verilog_code = instantiation_pattern.sub(new_name, verilog_code)

    return verilog_code, rename_map
