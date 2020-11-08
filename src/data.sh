#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --mem=300GB
#SBATCH --job-name=data_balancing
#SBATCH --mail-user=nsk367@nyu.edu
#SBATCH --output=slurm_%j.out

python dataloader.py