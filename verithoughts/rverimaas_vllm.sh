#!/usr/bin/env bash

rbatchsize=10
rnumsamples=1
rmodel="Qwen/Qwen2.5-14B"

# VeriThought
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op Generate --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op GenerateCoT --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op ReAct --self_refine

# VerilogEval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op Generate --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op GenerateCoT --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op ReAct --verilogeval --self_refine