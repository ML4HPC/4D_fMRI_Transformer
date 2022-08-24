#!/bin/bash
#SBATCH -A m3898_g
#SBATCH -J tr_n2
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:30:00
#SBATCH -N 2
#SBATCH -c 10
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --chdir=../
#SBATCH --exclusive
#SBATCH --account m3898_g
#SBATCH --output=slurm_logs/R-%x-%j.out
#SBATCH --mail-user=kjb961013@snu.ac.kr
set +x

# -c, --cpus-per-task

# -n, --ntasks=<number>
# Specify the number of tasks to run. Request that srun allocate resources for ntasks tasks. The default is one task per node, but note that the --cpus-per-task option will change this default. This option applies to job and step allocations.

# --ntasks-per-node
#Request that ntasks be invoked on each node. If used with the --ntasks option, the --ntasks option will take precedence and the --ntasks-per-node will be treated as a maximum count of tasks per node. Meant to be used with the --nodes option.


### init virtual environment if needed
#module load pytorch
source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
conda activate 3DCNN


### print slurm environment variables
env | grep SLURM

# Perlmutter
export SCRATCH=$SCRATCH

#torchrun will determine WORLD_SIZE #${SLURM_NNODES} # --rdzv_id ${SLURM_JOB_ID} 
#srun torchrun --nnodes=${SLURM_NNODES} --nproc_per_node=${SLURM_GPUS_PER_NODE} --rdzv_id ${SLURM_JOB_ID}  main.py --image_path /pscratch/sd/s/stella/ABCD_TFF/MNI_to_TRs --dataset_name ABCD --batch_size_phase1 64 --lr_init_phase1 1e-4 --lr_policy_phase1 SGDR --lr_warmup_phase1 500 --lr_gamma_phase1 0.5 --lr_step_phase1 500

srun main.py --image_path /pscratch/sd/s/stella/ABCD_TFF/MNI_to_TRs --dataset_name ABCD --batch_size_phase1 64 --lr_init_phase1 1e-4 --lr_policy_phase1 SGDR --lr_warmup_phase1 500 --lr_gamma_phase1 0.5 --lr_step_phase1 500
