SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. 
This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. 

In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution.

Do not include any additional text or explanation in the "solution_letter" field.
"""


REFLECTION_ON_PUBLIC_TEST_PROMPT = """
Given a code problem and a Verilog code solution which failed to pass test or execute with yosys, 
you need to analyze the reason for the failure and propose a better code solution: 

### problem
{problem}

### Code Solution
{solution}

### Yosys Execution Result
{exec_pass}

#### Yosys Failed Error Message
{test_fail}

Please provide a reflection on the failed test cases and code solution, 
followed by a better code solution without any additional text or test cases.
"""

SELFREFINE_PROMPT = """
You are an assistant specialized in refining solutions to problems.

Problem:
{problem}

Solution:
{solution}

Instruction:
Analyze the above solution for any errors or suboptimal aspects. 
Make iterative improvements to enhance its correctness and efficiency. Provide the refined solution below.
"""
