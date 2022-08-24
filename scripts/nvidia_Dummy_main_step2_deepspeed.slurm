#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH -t 4:00:00
#SBATCH --chdir=../
set +x

# -c, --cpus-per-task

# -n, --ntasks=<number>
# Specify the number of tasks to run. Request that srun allocate resources for ntasks tasks. The default is one task per node, but note that the --cpus-per-task option will change this default. This option applies to job and step allocations.

# --ntasks-per-node
#Request that ntasks be invoked on each node. If used with the --ntasks option, the --ntasks option will take precedence and the --ntasks-per-node will be treated as a maximum count of tasks per node. Meant to be used with the --nodes option.

singularity run --nv /home/be62tdqc/test.sif

sleep 10
export OMPI_MCA_btl_openib_warn_default_gid_prefix=0
export OMPI_MCA_btl_openib_cpc_exclude=rdmacm
export OMPI_MCA_btl_openib_if_exclude=mlx5_5:1,mlx5_11

env | grep SLURM

# load modules and conda
export NCCL_SOCKET_IFNAME=hsn

srun -u bash -c "
source export_DDP_vars.sh 
nsys profile -t cuda,nvtx,cublas,cudnn -f true -o output_%p /home/be62tdqc/.local/bin/deepspeed main_deepspeed.py --image_path /pscratch/sd/s/stella/ABCD_TFF/MNI_to_TRs --step 2 --dataset_name Dummy --profiling --batch_size_phase2 4 --random_TR --lr_init_phase2 1e-4 --lr_policy_phase2 SGDR --lr_warmup_phase2 500 --lr_gamma_phase2 0.5 --lr_step_phase2 500 --exp_name SGDR --deepspeed --deepspeed_config ds_config.json --exp_name deepspeed_dummy
" 
#nsys profile -t cuda,nvtx,cublas,cudnn -f true -o output_%p /home/be62tdqc/.local/bin/deepspeed  main_deepspeed.py --image_path /pscratch/sd/s/stella/ABCD_TFF/MNI_to_TRs --step 2 --dataset_name Dummy --profiling --batch_size_phase2 4 --random_TR --lr_init_phase2 1e-4 --lr_policy_phase2 SGDR --lr_warmup_phase2 500 --lr_gamma_phase2 0.5 --lr_step_phase2 500 --exp_name SGDR --deepspeed --deepspeed_config ds_config.json --exp_name deepspeed_dummy
