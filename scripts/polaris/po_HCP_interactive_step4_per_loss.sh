#!/bin/bash
cd ../..
module load conda
conda activate 3DCNN
torchrun --nnodes=1 --nproc_per_node=4 main_Stella.py --which_perceptual densenet3d --dataset_split_num 1 --step 4 --batch_size_phase4 1 --exp_name per_loss --init_method env --model_weights_path_phase3 '/lus/grand/projects/STlearn/4D_fMRI_Transformer/experiments/S1200_fine_tune_binary_classification_sex_per_loss/S1200_fine_tune_binary_classification_sex_per_loss_epoch_7_batch_index_559_BEST_val_AUROC.pth' --image_path '/lus/grand/projects/STlearn/HCP_MNI_to_TRs/'