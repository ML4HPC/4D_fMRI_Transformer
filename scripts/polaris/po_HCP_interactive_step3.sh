#!/bin/bash
cd ../..
#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
module load conda
conda activate 3DCNN
pip install torchio
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize_no_nan --exp_name voxel_normalize_no_nan_test --input_channel 2 
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize --exp_name voxel_normalize_nan_test --input_channel 2 
torchrun --nnodes=1 --nproc_per_node=4 main.py --num_workers 16 --dataset_split_num 2 --step 3 --lr_policy_phase3 SGDR --lr_init_phase3 1e-3 --lr_gamma_phase3 0.1  --batch_size_phase3 4 --exp_name polaris_job_interactive_split2_high_lr_ --init_method env --image_path /lus/grand/projects/STlearn/HCP_MNI_to_TRs/
 
