#!/bin/bash
cd ../..
#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
module load conda
conda activate 3DCNN
pip install torchio
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize_no_nan --exp_name voxel_normalize_no_nan_test --input_channel 2 
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize --exp_name voxel_normalize_nan_test --input_channel 2 
torchrun --nnodes=1 --nproc_per_node=4 main.py --which_perceptual densenet3d --dataset_split_num 1 --step 4 --batch_size_phase4 1 --dataset_name ABCD --exp_name per_loss --init_method env --model_weights_path_phase3 /lus/grand/projects/STlearn/4D_fMRI_Transformer/experiments/S1200_fine_tune_binary_classification_sex_per_loss/S1200_fine_tune_binary_classification_sex_per_loss_epoch_7_batch_index_559_BEST_val_AUROC.pth --image_path --image_path /lus/grand/projects/STlearn/8.masked_image_MNI_to_TRs

