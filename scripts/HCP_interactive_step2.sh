#!/bin/bash
cd ..
source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
conda activate 3DCNN
torchrun main.py --step 2 --batch_size_phase2 4 --model_weights_path_phase1 /global/cfs/cdirs/m3898/4D_fMRI_Transformer/experiments/S1200_autoencoder_reconstruction_sex_baseline/S1200_autoencoder_reconstruction_sex_baseline_epoch_7_batch_index_879_BEST_val_loss.pth --exp_name interactive-test
