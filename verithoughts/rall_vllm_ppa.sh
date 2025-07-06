#!/usr/bin/env bash

rbatchsize=10
rnumsamples=20
rmodel="Qwen/Qwen2.5-14B"

python3 generate_allops.py --prompt_op GenerateCoT --num_samples 20 --batch_size 10 --self_refine --ppa_op --model_name Qwen/Qwen2.5-14B --use_vllm
python3 generate_allops.py --prompt_op GenerateCoT --num_samples 20 --batch_size 10 --self_refine --ppa_op --model_name Qwen/Qwen2.5-14B --use_vllm --verilogeval 

# python3 evaluation_yosys.py --prompt_op GenerateCoT --num_samples 20 --batch_size 10 --self_refine --ppa_op --model_name Qwen/Qwen2.5-14B --use_vllm
# python3 evaluation_yosys.py --prompt_op GenerateCoT --num_samples 20 --batch_size 10 --self_refine --ppa_op --model_name Qwen/Qwen2.5-14B --use_vllm --verilogeval 

# python3 evaluation_ppa.py --prompt_op GenerateCoT --num_samples 20 --batch_size 10 --self_refine --ppa_op --model_name Qwen/Qwen2.5-14B --use_vllm
# python3 evaluation_ppa.py --prompt_op GenerateCoT --num_samples 20 --batch_size 10 --self_refine --ppa_op --model_name Qwen/Qwen2.5-14B --use_vllm --verilogeval 

