#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=72:00:00
##SBATCH --gres=gpu:k80:1
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --job-name=test_cube
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out

python test_cube.py
