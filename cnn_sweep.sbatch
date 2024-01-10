#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --gres=gpu:1
#SBATCH -o cnn_test.out
#SBATCH -e cnn_test.err
#SBATCH -q mcml
#SBATCH -t 3-12:00:00


# Set the path to your virtual environment's activate script
VENV_PATH=.venv/bin/activate

# Activate the virtual environment
source $VENV_PATH


srun python cnn_sweep.py