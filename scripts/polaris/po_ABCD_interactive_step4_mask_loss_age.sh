#!/bin/bash
cd ../..
module load conda
conda activate 3DCNN
torchrun --nnodes=1 --nproc_per_node=4 main_Stella.py --dataset_name ABCD --fine_tune_task regression --target age --dataset_split_num 1 --step 4 --batch_size_phase4 4 --exp_name mask_loss_1e4 --init_method env --model_weights_path_phase3 /lus/grand/projects/STlearn/4D_fMRI_Transformer/experiments/ABCD_fine_tune_regression_age_mask_loss_1e4/ABCD_fine_tune_regression_age_mask_loss_1e4_epoch_9_batch_index_333_BEST_val_loss.pth --image_path /lus/grand/projects/STlearn/8.masked_image_MNI_to_TRs