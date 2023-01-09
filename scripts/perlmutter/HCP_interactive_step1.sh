#!/bin/bash
cd ..
source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
conda activate 3DCNN

torchrun --nnodes=1 --nproc_per_node=4 main.py --batch_size_phase1 4 --exp_name voxel_norm 