#!/bin/bash
#BSUB -J task7_8
#BSUB -q hpc
#BSUB -W 4
#BSUB -R "rusage[mem=64MB]"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -o task7_8_%J.out
#BSUB -e task7_8_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python simulate_task5.py --n 20 --workers 8