import re, json, os
import numpy as np
from math import comb


def load_jsonl_file(filename): # from original repo
    des_data = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            des_data.append(data)
    return des_data

def savefile(filename, content, fmode='w'):
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


def extract_code_block_depr(text): # THEIR CODE: BUGGY! FAILS to catch ```verilog``` fences!!
    if not text:
        return ""
    pattern = r"CODE BEGIN(.*?)CODE END"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        s = matches[-1]  # take the last match
    else:
        s = ""
    return s


# this adapts our ActionNode MaAS sanitizer (duplicate here! TODO: Link-up at some point!)
# maas.actions.action_node: def verilog_fill
def extract_code_block(content: str) -> str:
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


def extract_modules(verilog_text):
    # This regex matches everything from 'module' to the nearest 'endmodule'
    pattern = r'\bmodule\b.*?\bendmodule\b'
    matches = re.findall(pattern, verilog_text, re.DOTALL)
    # Combine them into a single string, each separated by a newline
    combined_modules = '\n\n'.join(matches)
    return combined_modules

def parsing_helper(verilog_text, reasoning_mode):
    if reasoning_mode:
        parse1 = extract_code_block(verilog_text)
        if parse1 == '':
            parse1 = extract_modules(verilog_text)
            if parse1 == '':
                parse1 = verilog_text
    else:
        parse1 = extract_modules(verilog_text)
        if parse1 == '':
            parse1 = verilog_text
    return parse1


def load_json(filename):
    des_data = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            des_data.append(data)
    return des_data

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            item = json.loads(line.strip())
            data.append(item)
    return data

def pass_at_k(c_list, n, k):
    pass_at_k_values = []
    for c in c_list:
        if c == 0:
            pass_at_k_values.append(0.0)
        else:
            if n == k:
                pass_at_k_values.append(1.0 if c > 0 else 0.0)
            else:
                val = 1 - comb(n - c, k) / comb(n, k) if (n - c) >= k else 1.0
                pass_at_k_values.append(val)
    return np.mean(pass_at_k_values)


# Function to retrieve a specific result entry by q_id and sample_id
def get_result_entry(results, q_id, sample_id):
    """
    Fetches the result dict for the given (q_id, sample_id) pair.
    Raises ValueError if not found.
    """
    for entry in results:
        if entry.get("q_id") == q_id and entry.get("sample_id") == sample_id:
            return entry
    # If we get here, no matching entry was found
    raise ValueError(f"No result found for q_id={q_id}, sample_id={sample_id}")

