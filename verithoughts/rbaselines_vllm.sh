#!/usr/bin/env bash

rbatchsize=20 # default so no needed
rnumsamples=1
rmodel="Qwen/Qwen2.5-14B"

# VeriThought
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --use_vllm --prompt_op Generate
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --use_vllm --prompt_op GenerateCoT
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --use_vllm --prompt_op ReAct

# VerilogEval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --use_vllm --prompt_op Generate --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --use_vllm --prompt_op GenerateCoT --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --use_vllm --prompt_op ReAct --verilogeval