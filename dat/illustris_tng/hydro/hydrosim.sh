#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=72:00:00
#SBATCH --mem=180GB
#SBATCH --job-name=hsim
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out

python hydrosim.py
