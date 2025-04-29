#!/bin/bash
#BSUB -J task5_16
#BSUB -q hpc
#BSUB -W 4
#BSUB -R "rusage[mem=64MB]"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -o task5_16_50_%J.out
#BSUB -e task5_16_50_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python simulate_task5.py --n 50 --workers 16