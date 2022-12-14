#!/bin/bash
cd ../..
#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
module load conda
conda activate 3DCNN
# pip install torchio
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize_no_nan --exp_name voxel_normalize_no_nan_test --input_channel 2 
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize --exp_name voxel_normalize_nan_test --input_channel 2 
torchrun --nnodes=1 --nproc_per_node=4 main.py --dataset_split_num 1 --step 3 --batch_size_phase3 2 --exp_name con_loss --init_method env --use_cont_loss True --image_path /lus/grand/projects/STlearn/HCP_MNI_to_TRs/ --use_cont_loss True --dataset_split_num 1 --model_weights_path_phase2 './experiments/S1200_transformer_reconstruction_sex_con_loss/S1200_transformer_reconstruction_sex_con_loss_epoch_14_batch_index_1118_BEST_val_loss.pth'
 
