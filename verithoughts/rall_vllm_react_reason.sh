#!/usr/bin/env bash

rbatchsize=10
rnumsamples=20
rmodel="Qwen/Qwen3-14B"

# VeriThought
# python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op Generate --vllm_reasoning
# #Rnd2: python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op GenerateCoT
# #Lean: python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op ReAct  # Next: skipping ReAct

# VerilogEval
# python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op Generate --verilogeval --vllm_reasoning
# #Rnd2an: python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op GenerateCoT --verilogeval
# #Lean: python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op ReAct --verilogeval  # Next: skipping ReAct

# VeriThought VeriMaAS
# #Rnd2: python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op Generate --self_refine
# python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op GenerateCoT --self_refine --vllm_reasoning
# #Lean: 
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op ReAct --self_refine --vllm_reasoning --resume # Now! # Next: skipping ReAct

# VerilogEval VeriMaAS
# #Rnd2: python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op Generate --verilogeval --self_refine
# python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op GenerateCoT --verilogeval --self_refine --vllm_reasoning
# 
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op ReAct --verilogeval --self_refine --vllm_reasoning  # Next: skipping ReAct