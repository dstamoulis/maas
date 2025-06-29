#!/usr/bin/env bash

rbatchsize=20

rmodel="gpt-4o-mini"
# gpt-4o-mini c=1
rnumsamples=1
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval --self_refine
# gpt-4o-mini c=20
rnumsamples=20
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval --self_refine


rmodel="o4-mini"
# gpt-4o-mini c=1
rnumsamples=1
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval --self_refine
# gpt-4o-mini c=20
rnumsamples=20
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval --self_refine



rmodel="o3"
# gpt-4o-mini c=1
rnumsamples=1
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval --self_refine
# gpt-4o-mini c=20
rnumsamples=20
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op Generate --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op GenerateCoT --verilogeval --self_refine
python3 generate_allops.py --model_name $rmodel --num_samples $rnumsamples --batch_size $rbatchsize --prompt_op ReAct --verilogeval --self_refine