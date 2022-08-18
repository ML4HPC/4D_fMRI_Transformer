#!/bin/bash
cd ..
source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
conda activate 3DCNN
# by specifying --nnodes and --nproc_per_node, it set up the env variable
torchrun --nnodes=1 --nproc_per_node=4  main.py --image_path /pscratch/sd/s/stella/ABCD_TFF/MNI_to_TRs --dataset_name ABCD --batch_size_phase1 64 --random_TR --lr_init_phase1 1e-4 --lr_policy_phase1 SGDR --lr_warmup_phase1 500 --lr_gamma_phase1 0.5 --lr_step_phase1 500 --exp_name SGDR