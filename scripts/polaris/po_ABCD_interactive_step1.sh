#!/bin/bash
cd ../..
#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
module load conda
conda activate 3DCNN
#pip install torchio
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize_no_nan --exp_name voxel_normalize_no_nan_test --input_channel 2 
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize --exp_name voxel_normalize_nan_test --input_channel 2 
torchrun --nnodes=1 --nproc_per_node=4 main.py --step 1 --init_method file --batch_size_phase1 4  --use_intensity_loss --dataset_name ABCD --exp_name interactive  --image_path /lus/grand/projects/STlearn/7.cleaned_image_MNI_to_TRs --dataset_split_num 1
# --with_voxel_norm
