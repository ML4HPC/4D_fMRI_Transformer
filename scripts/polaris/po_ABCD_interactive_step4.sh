#!/bin/bash
cd ../..
#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
module load conda
conda activate 3DCNN
pip install torchio
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize_no_nan --exp_name voxel_normalize_no_nan_test --input_channel 2 
#torchrun --nnodes=1 --nproc_per_node=4 main.py --step 3 --batch_size_phase3 8 --voxel_norm_dir per_voxel_normalize --exp_name voxel_normalize_nan_test --input_channel 2 
# torchrun --nnodes=1 --nproc_per_node=4 main.py --dataset_name ABCD --fine_tune_task binary_classification --target ADHD --num_workers 16 --dataset_split_num 1 --step 4 --batch_size_phase4 4 --exp_name ABCD_cleaned_image_ADHD_split1 --image_path /lus/grand/projects/STlearn/7.cleaned_image_MNI_to_TRs --model_weights_path_phase3 ./experiments/ABCD_fine_tune_binary_classification_ASD_ABCD_cleaned_image_ADHD_split1/ABCD_fine_tune_binary_classification_ASD_ABCD_cleaned_image_ADHD_split1_epoch_2_batch_index_269_BEST_val_AUROC.pth
torchrun --nnodes=1 --nproc_per_node=4 main.py --step 4 --dataset_name ABCD --fine_tune_task regression --target age --num_workers 16 --dataset_split_num 1 --batch_size_phase4 4 --exp_name ABCD_cleaned_image_regression_split1 --image_path /lus/grand/projects/STlearn/7.cleaned_image_MNI_to_TRs --model_weights_path_phase3 /lus/grand/projects/STlearn/4D_fMRI_Transformer/experiments/ABCD_fine_tune_regression_age_ABCD_cleaned_image_regression_split1/ABCD_fine_tune_regression_age_ABCD_cleaned_image_regression_split1_epoch_2_batch_index_333_BEST_val_loss.pth

