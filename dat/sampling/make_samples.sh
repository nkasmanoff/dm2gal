#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=800GB
#SBATCH --job-name=sampling
#SBATCH --mail-type=END
#SBATCH --output=slurm_%j.out

python make_samples.py

