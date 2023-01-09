#!/bin/bash
cd ..
source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
conda activate 3DCNN
#DP
# /pscratch/sd/s/stella/ABCD_TFF/MNI_to_TRs
#torchrun
torchrun main.py --image_path /pscratch/sd/s/stella/ABCD_TFF_20_timepoint_removed/MNI_to_TRs --step 2 --init_method env --dataset_name ABCD --batch_size_phase2 4 --lr_policy_phase2 SGDR --lr_init_phase2 1e-4 --lr_warmup_phase2 500 --lr_step_phase2 1000 --model_weights_path_phase1 /global/cfs/cdirs/m3898/4D_fMRI_Transformer/experiments/ABCD_autoencoder_reconstruction_sex_v1/ABCD_autoencoder_reconstruction_sex_v1_epoch_11_batch_index_2134_BEST_val_loss.pt --exp_name interactive-test

#torch launcher
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --image_path /pscratch/sd/s/stella/ABCD_TFF/MNI_to_TRs --step 2 --dataset_name ABCD --batch_size_phase2 4 --lr_policy_phase2 SGDR --lr_init_phase2 1e-4 --lr_warmup_phase2 500 --lr_step_phase2 1000 --model_weights_path_phase1 /global/cfs/cdirs/m3898/4D_fMRI_Transformer/experiments/ABCD_autoencoder_reconstruction_sex_baseline/ABCD_autoencoder_reconstruction_sex_baseline_epoch_9_batch_index_2205_BEST_val_loss.pth --exp_name interactive-test