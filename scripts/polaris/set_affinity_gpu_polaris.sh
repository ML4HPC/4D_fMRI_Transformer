#!/bin/bash
#num_gpus=$(nvidia-smi -L | wc -l)
#gpu=$((${PMI_LOCAL_RANK} % ${num_gpus}))
#export CUDA_VISIBLE_DEVICES=$gpu

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=$(nvidia-smi -L | wc -l)

export WORLD_SIZE=$(( NNODES * NRANKS_PER_NODE ))
export RANK=${PMI_RANK}
export LOCAL_RANK=${PMI_LOCAL_RANK}
export MASTER_PORT=29500
echo “WORLD_SIZE= ${WORLD_SIZE} RANK= ${RANK} LOCAL_RANK= ${LOCAL_RANK}”
exec "$@"
