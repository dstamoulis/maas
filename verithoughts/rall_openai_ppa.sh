#!/usr/bin/env bash

rbatchsize=10
rnumsamples=20

# rmodel="gpt-4o-mini"
python3 generate_allops.py --prompt_op GenerateCoT --num_samples 20 --self_refine --ppa_op --model_name o4-mini
python3 generate_allops.py --prompt_op GenerateCoT --num_samples 20 --self_refine --ppa_op --model_name o4-mini --verilogeval 

python3 evaluation_yosys.py --prompt_op GenerateCoT --num_samples 20 --self_refine --ppa_op --model_name o4-mini
python3 evaluation_yosys.py --prompt_op GenerateCoT --num_samples 20 --self_refine --ppa_op --model_name o4-mini --verilogeval 

python3 evaluation_ppa.py --prompt_op GenerateCoT --num_samples 20 --self_refine --ppa_op --model_name o4-mini
python3 evaluation_ppa.py --prompt_op GenerateCoT --num_samples 20 --self_refine --ppa_op --model_name o4-mini --verilogeval 


# Needed for GPT-4o mini c=20 VerilogEval!
python3 evaluation_yosys.py --prompt_op GenerateCoT --num_samples 20 --self_refine --ppa_op --verilogeval
python3 evaluation_ppa.py --prompt_op GenerateCoT --num_samples 20 --self_refine --verilogeval