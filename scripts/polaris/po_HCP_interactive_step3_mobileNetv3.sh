#!/bin/bash
cd ../..
#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
module load conda
conda activate 3DCNN
# pip install torchio
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize_no_nan --exp_name voxel_normalize_no_nan_test --input_channel 2 
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize --exp_name voxel_normalize_nan_test --input_channel 2 
#torchrun --nnodes=1 --nproc_per_node=4 main.py --block_type MobileNet_v3 --dataset_split_num 1 --step 3 --batch_size_phase3 16 --exp_name polaris_job_test_mobilenet_v3 --image_path /lus/grand/projects/STlearn/HCP_MNI_to_TRs/
torchrun --nnodes=1 --nproc_per_node=4 main.py --use_optuna --block_type MobileNet_v3 --dataset_split_num 1 --step 3 --batch_size_phase3 16 --exp_name polaris_job_test_mobilenet_v3 --image_path /lus/grand/projects/STlearn/HCP_MNI_to_TRs/ #--model_weights_path_phase3 /lus/grand/projects/STlearn/4D_fMRI_Transformer/experiments/S1200_fine_tune_binary_classification_sex_mobilenetv3_from_scratch_split_1_bs16_Adam_dropout/S1200_fine_tune_binary_classification_sex_mobilenetv3_from_scratch_split_1_bs16_Adam_dropout_epoch_13_batch_index_69_BEST_val_AUROC.pth
