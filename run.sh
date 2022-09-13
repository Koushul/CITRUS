#!/bin/bash
#SBATCH -N 1
#SBATCH -t 6-00:00
#SBATCH --cluster=htc
#SBATCH --mem=256g
#SBATCH --cpus-per-task=8
#SBATCH --output=integrate.log
#SBATCH --job-name=Integrate

module load python
workon deepbio

python integrate.py

