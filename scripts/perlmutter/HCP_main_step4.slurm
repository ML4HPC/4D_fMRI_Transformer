#!/bin/bash
#SBATCH -A m3898_g
#SBATCH -J TFF_HCP_step_one
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -N 4
#SBATCH -c 10
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --chdir=../
#SBATCH --exclusive
#SBATCH --account m3898_g
#SBATCH --output=slurm_logs/R-%x-%j.out
#SBATCH --mail-user=kjb961013@snu.ac.kr
#SBATCH --mail-type=BEGIN
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

# Phase4 : test
# have some problems here - Stella will solve it! don't worry
#srun python main.py --step test #--model_weights_path_phase3 ./weights/S1200_fine_tune_binary_classification.pth

python main.py --step 4 --exp_name test --image_path /dvs_ro/cfs/cdirs/m3898/4D_fMRI_Transformer/MNI_to_TRs --init_method env --voxel_norm_dir per_voxel_normalize_no_nan --model_weights_path_phase3 ./experiments/S1200_fine_tune_binary_classification_sex_voxel_normalize_min/S1200_fine_tune_binary_classification_sex_voxel_normalize_min_epoch_8_batch_index_349_BEST_val_loss.pth