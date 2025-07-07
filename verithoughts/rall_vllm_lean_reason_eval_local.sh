#!/usr/bin/env bash

rbatchsize=10
rnumsamples=20
rmodel="Qwen/Qwen3-14B"

# VeriThought
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op Generate --vllm_reasoning
# #Rnd2: python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op GenerateCoT
# #Lean: python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op ReAct  # Next: skipping ReAct

# VerilogEval
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op Generate --verilogeval --vllm_reasoning
# #Rnd2an: python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op GenerateCoT --verilogeval
# #Lean: python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op ReAct --verilogeval  # Next: skipping ReAct

# VeriThought VeriMaAS
# #Rnd2: python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op Generate --self_refine
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op GenerateCoT --self_refine --vllm_reasoning --ppa_op
# #Lean: python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op ReAct --self_refine # Now! # Next: skipping ReAct

# VerilogEval VeriMaAS
# #Rnd2: python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op Generate --verilogeval --self_refine
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op GenerateCoT --verilogeval --self_refine --vllm_reasoning --ppa_op
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --use_vllm --prompt_op ReAct --verilogeval --self_refine  # Next: skipping ReAct