#!/usr/bin/env bash

rbatchsize=10
rnumsamples=20
rmodel="Qwen/Qwen2.5-14B"

# VeriThought
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op Generate
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op GenerateCoT
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op ReAct

# VerilogEval
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op Generate --verilogeval
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op GenerateCoT --verilogeval
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op ReAct --verilogeval