#!/bin/bash
cd ..
source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
conda activate 3DCNN
module load cudatoolkit
CUDA_VISIBLE_DEVICES=0,1,2,3

#phase1
#nsys profile -t cuda,nvtx,cublas,cudnn -f true -o output_%p python -m torch.distributed.launch --nproc_per_node=4 main.py --profiling --step 1 --batch_size_phase1 8

#phase2
nsys profile -t cuda,nvtx,cublas,cudnn -f true -o output_%p python -m torch.distributed.launch --nproc_per_node=4 main.py --profiling --step 2 --batch_size_phase2 4 --model_weights_path_phase1 /global/cfs/cdirs/m3898/4D_fMRI_Transformer/experiments/S1200_autoencoder_reconstruction_07_27__09_42_36/S1200_autoencoder_reconstruction_07_27__09_42_36_epoch_0_batch_index_28155_BEST_val_loss.pth