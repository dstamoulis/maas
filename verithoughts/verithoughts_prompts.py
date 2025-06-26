
INSTR_SIMPLE = "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.\n" 

INSTR_REASONING = "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.<think>\n"

REFLECTION_ON_YOSYS_TEST_PROMPT = """
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