IMPROVE_CODE_PROMPT = """
The previous solution failed some test cases in the VeriThoughts benchmark. Please conduct a thorough analysis of the problem statement, 
identifying all edge cases and potential pitfalls. Then, provide an improved solution that not only fixes the issues but also optimizes 
performance and adheres to industry-standard coding practices. Ensure your revised code includes clear, concise comments that explain your 
logic and design choices, and that it robustly handles all specified requirements.
"""

GENERATE_CODE_PROMPT = """
You are an expert programmer with deep knowledge in Verilog design and code optimization. Your task is to generate a clear, efficient, and 
well-documented solution to the problem described above, as presented in the VeriThoughts dataset. Please include all relevant code snippets 
along with detailed explanations of your reasoning, the algorithms used, and how you handle edge cases. Ensure that your solution is easy to understand, 
follows best practices, and is structured to facilitate future maintenance and enhancements.
"""

INSTR_SIMPLE = "Make sure your input and output interface has the same names as described in the question. \nPlease start your Verilog code with CODE BEGIN and end with CODE END.\n" 
