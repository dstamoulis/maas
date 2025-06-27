
INSTR_SIMPLE = "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.\n" 

INSTR_REASONING = "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.<think>\n"

SELFREFINE_PROMPT_FROM_YOSYS_TEST = """
Given a code task and a Verilog code solution which failed to execute with yosys, 
you need to analyze the reason for the failure and propose a better code solution: 

### Code Task
{code_task}

### Erroneous Code Solution
{solution}

#### Yosys Failed Error Message
{test_fail}

Please reflect on the failed test cases and code solution to understand the root of the issue.
Based on this, your task is to provide new corrected code solution that addresses the issue.

Make sure your input and output interface has the same names as described in the question. 
Please start your new corrected Verilog code solution with CODE BEGIN and end with CODE END.

"""

SC_ENSEMBLE_PROMPT = """
Given a Verilog code task and a set of solution which have been generated to solve the given task, 
you need to analyze them and identify the solution that appearts most frequently across them.

### Code Task
{code_task}

### Code Solutions:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. 
This consistency in answers is crucial for determining the most reliable solution.

First, reflect in detail what is your thought process. Based on this, your task is to 
output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution.

Make sure your choice is a letter ID present in provided solution options.
Please reply with the LETTER choice ONLY!
Do not include any additional text or explanation!

"""


REACT_PROMPT = """

{code_task}

Make sure your input and output interface has the same names as described in the question. 
Please start your Verilog code with CODE BEGIN and end with CODE END.\n 

Some tips if helpful!
 
* Might be useful to understand fully whether the task requires combinational or sequential logic. 

* Also, always good to double-check that the interface (inputs/outputs) in your solution matches the names and bit widths described in the task.

You are the best Verilog designer I know pal! You can do this =)

"""


DECIDE_PROMPT = """
You will be given a block of LLM reasoning that ends with a choice recommendation. This is from 
a previous call where the agent was presented with different solutions A, B, ... and reasons which
on is the more correct one.

### Reasoning trace:
{reasoning_track}

Your job is to extract the final choice from this reasoning.  
- The choice must be one of the letter IDs (A, B, C, etc.) corresponding to the provided options.  
- Output only the single letter (e.g. “A”), with no surrounding punctuation, explanation, or extra text.

Make sure your choice is a letter ID present in provided reasoning.
Please reply with the LETTER choice ONLY!
Do not include any additional text or explanation!

"""
