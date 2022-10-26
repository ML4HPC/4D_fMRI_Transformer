#!/bin/bash
cd ..
#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
#conda activate 3DCNN
#module load cudatoolkit
#CUDA_VISIBLE_DEVICES=0,1,2,3

export SCRATCH=/tmp
#phase1
#nsys profile -t cuda,nvtx,cublas,cudnn -f true -o output_%p torchrun --nnodes=1 --nproc_per_node=1 main.py --profiling --step 1 --workers_phase1 4 --batch_size_phase1 8 --dataset_name Dummy 

#nsys profile -t cuda,nvtx,cublas,cudnn -f true -o output_%p python -m torch.distributed.launch --nproc_per_node=2  main.py --profiling --step 1 --workers_phase1 4 --batch_size_phase1 8 --dataset_name Dummy

#phase2
nsys profile -t cuda,nvtx,cublas,cudnn -f true -o output_%p torchrun --nnodes=1 --nproc_per_node=1  main.py --profiling --step 2 --workers_phase2 4 --amp  --batch_size_phase2 4 --dataset_name Dummy 