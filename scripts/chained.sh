#!/bin/bash
# Run 2 consecutive jobs of cycleGAN

iterations=1 # 총 몇 번이나 연속으로 돌릴 것인지
jobid=$(sbatch --parsable /global/cfs/cdirs/m3898/GANBERT/t1_harmonization/3D-cycleGAN/scripts/run_main.slurm)

for((i=0; i<$iterations; i++)); do            
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency /global/cfs/cdirs/m3898/GANBERT/t1_harmonization/3D-cycleGAN/scripts/run_main.slurm)
    dependency=",${dependency}afterany:${jobid}"
done