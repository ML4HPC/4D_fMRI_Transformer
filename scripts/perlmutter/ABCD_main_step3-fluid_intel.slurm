#!/bin/bash
#SBATCH -A m3898_g
#SBATCH -J TFF_ABCD_step_three
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --chdir=../
#SBATCH --exclusive
#SBATCH --account m3898_g
#SBATCH --output=slurm_logs/R-%x-%j-fluid.out
#SBATCH --mail-user=kjb961013@snu.ac.kr
set +x

# -c, --cpus-per-task

# -n, --ntasks=<number>
# Specify the number of tasks to run. Request that srun allocate resources for ntasks tasks. The default is one task per node, but note that the --cpus-per-task option will change this default. This option applies to job and step allocations.

# --ntasks-per-node
#Request that ntasks be invoked on each node. If used with the --ntasks option, the --ntasks option will take precedence and the --ntasks-per-node will be treated as a maximum count of tasks per node. Meant to be used with the --nodes option.

#module load python
source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
#module load pytorch
conda activate 3DCNN

export MASTER_ADDR=$(hostname)

env | grep SLURM

srun bash -c "
source export_DDP_vars.sh  
python main.py --image_path /pscratch/sd/s/stella/ABCD_TFF_20_timepoint_removed/MNI_to_TRs --dataset_name ABCD --step 3 --batch_size_phase3 4 --exp_name v1 --target nihtbx_fluidcomp_uncorrected --fine_tune_task regression --workers_phase1 4"


#srun python main.py --image_path /pscratch/sd/s/stella/ABCD_TFF/MNI_to_TRs --dataset_name ABCD --step 3 --batch_size_phase3 4 --target nihtbx_fluidcomp_uncorrected --fine_tune_task regression --resume 
