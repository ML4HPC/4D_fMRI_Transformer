#!/bin/bash
# Run 3 consecutive jobs

iterations=3 # 총 몇 번이나 연속으로 돌릴 것인지 (일단 3으로 해놓음 그래야 내가 내일 와서 확인을 하지..)
jobid=$(sbatch --parsable /global/cfs/cdirs/m3898/TFF/main.slurm)

for((i=0; i<$iterations; i++)); do            
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency /global/cfs/cdirs/m3898/TFF/main.slurm)
    dependency=",${dependency}afterany:${jobid}"
done
