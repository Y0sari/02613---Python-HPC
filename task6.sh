#!/bin/bash
#BSUB -J task6_1
#BSUB -q hpc
#BSUB -W 5
#BSUB -R "rusage[mem=128MB]"
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -o task6_1_%J.out
#BSUB -e task6_1_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python simulate_task5.py --n 20 --workers 1