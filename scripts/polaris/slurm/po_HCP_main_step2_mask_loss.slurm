#!/bin/bash
#SBATCH -A m3898_g
#SBATCH -J H_s2_mask_loss
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --chdir=../..
#SBATCH --exclusive
#SBATCH --account m3898_g
#SBATCH --output=slurm_logs/R-%x-%j.out
#SBATCH --mail-user=stellasybae@snu.ac.kr
set +x

# -c, --cpus-per-task

# -n, --ntasks=<number>
# Specify the number of tasks to run. Request that srun allocate resources for ntasks tasks. The default is one task per node, but note that the --cpus-per-task option will change this default. This option applies to job and step allocations.

# --ntasks-per-node
#Request that ntasks be invoked on each node. If used with the --ntasks option, the --ntasks option will take precedence and the --ntasks-per-node will be treated as a maximum count of tasks per node. Meant to be used with the --nodes option.

module load conda
conda activate 3DCNN

env | grep SLURM

srun bash -c "
source export_DDP_vars.sh  
python main.py --step 2 --batch_size_phase2 4 --exp_name mask_loss_Stella_1024 --use_mask_loss True --image_path /lus/grand/projects/STlearn/HCP_MNI_to_TRs --workers_phase2 4 "
