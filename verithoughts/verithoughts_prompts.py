
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