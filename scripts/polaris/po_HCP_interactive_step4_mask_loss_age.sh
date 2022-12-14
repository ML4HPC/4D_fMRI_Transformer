#!/bin/bash
cd ../..
module load conda
conda activate 3DCNN
torchrun --nnodes=1 --nproc_per_node=4 main_Stella.py --dataset_split_num 1 --fine_tune_task regression --target age --step 4 --batch_size_phase4 4 --exp_name mask_loss --init_method env --model_weights_path_phase3 /lus/grand/projects/STlearn/4D_fMRI_Transformer/experiments/S1200_fine_tune_regression_age_mask_loss_1e4/S1200_fine_tune_regression_age_mask_loss_1e4_epoch_14_batch_index_279_BEST_val_AUROC.pth --image_path /lus/grand/projects/STlearn/HCP_MNI_to_TRs/
