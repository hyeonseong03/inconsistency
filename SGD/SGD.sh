#!/bin/bash
#
#SBATCH --job-name=SGD
#SBATCH --partition=intern 
#SBATCH --output=%A_%a_res.txt
#SBATCH --error=%A_%a_err.txt
#SBATCH --nodelist=lrning-mainserver
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=16G

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6

#SBATCH -t 30:00:00

hostname
eval "$(conda shell.bash hook)"
conda activate hyeonseongkim
python SGD.py

exit 0
