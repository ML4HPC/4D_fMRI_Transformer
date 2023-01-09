#!/bin/bash
cd ..
source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
conda activate 3DCNN
torchrun main.py --step 3 --batch_size_phase3 8 --target age --fine_tune_task regression