
GENERATE_PROMPT = "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.\n" 

GENERATE_COT_PROMPT = """

{code_task}

Make sure your input and output interface has the same names as described in the question. 
Please start your Verilog code with CODE BEGIN and end with CODE END.\n 

Some tips if helpful!
 
* Might be useful to understand fully whether the task requires combinational or sequential logic. 

* Also, always good to double-check that the interface (inputs/outputs) in your solution matches the names and bit widths described in the task.

You are the best Verilog designer I know pal! You can do this =)

"""

INSTR_REASONING = "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.<think>\n"

REFINE_PROMPT = """
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

Some tips if helpful!
 
* Might be useful to understand fully whether the task requires combinational or sequential logic. 

* Also, always good to double-check that the interface (inputs/outputs) in your solution matches the names and bit widths described in the task.

Make sure your input and output interface has the same names as described in the question. 
Please start your new corrected Verilog code solution with CODE BEGIN and end with CODE END.

"""

SC_ENSEMBLE_PROMPT = """
Given a Verilog code task and a set of solutions which have been generated to solve the given task, 
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

DEBATE_PROMPT = """
Given a Verilog code task and a set of candidate solutions that were generated for that task,  
your job is to analyze them and identify the one most likely to be incorrect or fail the specification.

### Code Task
{code_task}

### Candidate Solutions
{solutions}

First, reflect in detail on your thought process. Make sure to evaluate each candidate against the following points:

* **Combinational vs. Sequential Logic**: Does the task require purely combinational logic or an edge-triggered sequential block?  Verify that combinational designs use `assign` and that sequential designs use `always @(…)` with non-blocking assignments, without mixing styles.  
* **Fundamental Design Concepts**: Check that each solution correctly implements the required logic components or algorithms (multiplexers, adders, encoders, FSMs, etc.) and follows a clear structural outline.  
* **Register and Wire Widths**: Ensure that all `reg` and `wire` declarations match the bit widths specified by the task.  Do not default to `[31:0]` unless the spec demands it.  
* **Timing and Triggers**: Confirm that synchronous and asynchronous resets, clock edges, and enable signals are handled properly in sequential designs to meet the timing requirements.  
* **Truth-Table Coverage**: For combinational solutions, verify that all input combinations are covered and that no race conditions or latches are introduced.  
* **Interface Consistency**: The port names, directions, and widths must exactly match the task specification in every solution.

After your detailed reflection, output **only** the single letter ID (A, B, ...) corresponding to the candidate you judge **most likely to be incorrect**.  
Make sure your choice is one of the provided letter IDs and nothing else.  
Do **not** include any additional explanation or text—only the letter.
"""


DEBATE_PROMPT_SIMPLE = """
Given a Verilog code task and a set of candidate solutions that were generated for that task,  
your job is to analyze them and identify the one most likely to be incorrect or fail the specification.

### Code Task
{code_task}

### Candidate Solutions
{solutions}

First, reflect on your evaluation process against these clear criteria:

* **Logic Modality**: Determine whether the implementation is purely combinational or requires sequential elements, and check that each solution follows a consistent approach.  
* **Correct Use of Primitives**: Verify that each design uses the right fundamental constructs (such as adders, multiplexers, state machines) in a manner that aligns with the problem's requirements.  
* **Signal Declarations**: Ensure that all internal signals and registers are declared and sized appropriately for their intended roles, without introducing unintended storage or extra bits.  
* **Control and Timing**: Confirm that any clock or reset behavior is handled correctly so the design will operate as intended over time.  
* **Functional Coverage**: For purely combinational logic, check that every input scenario is addressed; for sequential logic, ensure all state transitions are defined.  
* **Interface Matching**: Confirm that every solution exactly matches the specified port names, directions, and signal roles in the original task.

After your detailed reflection, output **only** the single letter ID (A, B, ...) corresponding to the candidate you judge **most likely to be incorrect**.  
Make sure your choice is one of the provided letter IDs and nothing else.  
Do **not** include any additional explanation or text—only the letter.
"""


REACT_PROMPT_SIMPLE = """

Given a Verilog code task you need to analyze the requirements and propose a correct, synthesizable code solution:
 
### Code Task
{code_task}
 
In your response, be sure to address these clear criteria:

* **Logic Modality**: Identify whether the design is purely combinational or requires sequential behavior, and apply the appropriate construct consistently.  
* **Signal Declarations**: Declare and size all internal signals and registers appropriately for their intended use, avoiding unnecessary storage.  
* **Control Flow**: Handle any clocking and reset requirements so that the design behaves correctly over time.  
* **Functional Coverage**: For combinational logic, cover every input scenario; for sequential logic, define all state transitions.  
* **Interface Compliance**: Match the specified port names, directions, and signal roles exactly as given.

First, explain in detail how you will apply each of these points, if applicable!

Make sure your input and output interface has the same names as described in the question. 
Please start your Verilog code solution with CODE BEGIN and end with CODE END.

"""


REACT_PROMPT = """

Given a Verilog code task you need to analyze the requirements and propose a correct, synthesizable code solution:
 
### Code Task
{code_task}
 
In the response generation process, make sure to address the following points:
 
* **Combinational vs. Sequential Logic**: Determine whether the task requires combinational or sequential logic. Use `assign` for combinational and `always @('appropiate clock edge or signal name')` with non-blocking assignments for sequential logic. Avoid mixing them.
 
* **Fundamental Design Concepts**: Ensure a solid understanding of basic logic components and/or algorithms relevant to the problem (e.g., multiplexers, adders, encoders, FSMs). Structure your solution with a clear outline before writing code.
 
* **Register Widths**: Use register and wire widths that match the task specification. Don't default to `[31:0]` unless required.
 
* **Timing and Triggers**: Trigger actions on appropriate signals, including synchronous and asynchronous resets. Ensure correct timing behavior, especially across clock cycles in sequential designs.
 
* **Truth Table Coverage**: For combinational logic, ensure the implementation satisfies the full truth table. Consider all possible input cases in your design. Avoid race conditions in the final design. 
 
The interface (inputs/outputs) in your solution must match the names and bit widths described in the task.
 
First, present your reasoning in detail on how exactly you will be addressing these points, if applicable!

Make sure your input and output interface has the same names as described in the question. 
Please start your Verilog code solution with CODE BEGIN and end with CODE END.

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
