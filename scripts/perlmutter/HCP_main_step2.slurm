#!/bin/bash
#SBATCH -A m3898_g
#SBATCH -J H_s2
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
#SBATCH --output=slurm_logs/R-%x-%j.out
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

env | grep SLURM

export MASTER_ADDR=$(hostname)

srun bash -c "
source export_DDP_vars.sh  
python main.py --step 2 --batch_size_phase2 4 --exp_name from_step1_v2 --image_path /global/cfs/cdirs/m3898/4D_fMRI_Transformer/MNI_to_TRs --workers_phase2 4 --model_weights_path_phase1 /global/cfs/cdirs/m3898/4D_fMRI_Transformer/experiments/S1200_autoencoder_reconstruction_sex_v1/S1200_autoencoder_reconstruction_sex_v1_epoch_9_batch_index_703_BEST_val_loss.pth"

#--model_weights_path_phase1 #/global/cfs/cdirs/m3898/4D_fMRI_Transformer/experiments/S1200_autoencoder_reconstruction_sex_ba#seline/S1200_autoencoder_reconstruction_sex_baseline_epoch_7_batch_index_879_BEST_val_loss.pth
#"


