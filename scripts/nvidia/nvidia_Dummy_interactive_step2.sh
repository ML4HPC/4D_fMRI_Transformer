#!/bin/bash
cd ..
#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
#conda activate 3DCNN
#module load pytorch
#pip install dill
#pip install deepspeed
#pip install transformers

#export MASTER_ADDR=$(hostname)


nsys profile -t cuda,nvtx,cublas,cudnn -f true -o output_%p torchrun --nnodes=1 --nproc_per_node=4 main.py --image_path /pscratch/sd/s/stella/ABCD_TFF/MNI_to_TRs --step 2 --dataset_name Dummy --profiling --batch_size_phase2 4 --random_TR --lr_init_phase2 1e-4 --lr_policy_phase2 SGDR --lr_warmup_phase2 500 --lr_gamma_phase2 0.5 --lr_step_phase2 500 --exp_name SGDR  --exp_name deepspeed_dummy --sequence_length_phase2 59
