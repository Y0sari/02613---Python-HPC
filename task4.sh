#!/bin/bash
#BSUB -J task4
#BSUB -q hpc
#BSUB -W 4
#BSUB -R "rusage[mem=256MB]"
#BSUB -o task4_%J.out
#BSUB -e task4_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

kernprof -l -v simulate.py 20