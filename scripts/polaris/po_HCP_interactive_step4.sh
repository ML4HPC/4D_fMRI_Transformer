#!/bin/bash
cd ../..
#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
module load conda
conda activate 3DCNN
pip install torchio
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize_no_nan --exp_name voxel_normalize_no_nan_test --input_channel 2 
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize --exp_name voxel_normalize_nan_test --input_channel 2 
#torchrun --nnodes=1 --nproc_per_node=4 main.py --dataset_split_num 1 --step 4 --batch_size_phase4 4 --exp_name polaris_job --init_method env --model_weights_path_phase3 /lus/grand/projects/STlearn/4D_fMRI_Transformer/experiments/S1200_fine_tune_binary_classification_sex_from_scratch_split_1/S1200_fine_tune_binary_classification_sex_from_scratch_split_1_epoch_7_batch_index_279_BEST_val_accuracy.pth --image_path /lus/grand/projects/STlearn/HCP_MNI_to_TRs/
torchrun --nnodes=1 --nproc_per_node=4 main.py  --fine_tune_task regression --target age --num_workers 16 --dataset_split_num 1 --step 4 --batch_size_phase4 4 --exp_name split1 --image_path /lus/grand/projects/STlearn/HCP_MNI_to_TRs/ --model_weights_path_phase3 /lus/grand/projects/STlearn/4D_fMRI_Transformer/experiments/S1200_fine_tune_regression_age_from_scratch_age_split1/S1200_fine_tune_regression_age_from_scratch_age_split1_epoch_4_batch_index_279_BEST_val_loss.pth
