#!/usr/bin/env bash

rbatchsize=10
rnumsamples=20

rmodel="gpt-4o-mini"
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct # skipping ReAct
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval # skipping ReAct

python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --self_refine 
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --self_refine
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --self_refine # skipping ReAct
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval --self_refine
python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval --self_refine
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval --self_refine # skipping ReAct

# rmodel="o4-mini"
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT
# # # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct # skipping ReAct
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval
# # # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval # skipping ReAct

# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --self_refine 
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --self_refine
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --self_refine
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval --self_refine
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval --self_refine
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval --self_refine

# # rmodel="o3"
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct
# python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --self_refine
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --self_refine
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --self_refine
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval --self_refine
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval --self_refine
# # python3 evaluation_yosys.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval --self_refine

