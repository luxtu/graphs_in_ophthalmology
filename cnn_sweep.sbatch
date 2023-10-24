#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --gres=gpu:1
#SBATCH -o cnn_test.out
#SBATCH -e cnn_test.err


# Set your API key (replace 'YOUR_API_KEY' with your actual API key)
API_KEY= 6e4b754c329fb92daa33fb7ed96d9ef6db220d9f

# Log in to wandb using your API key
wandb login $API_KEY

# Set the path to your virtual environment's activate script
VENV_PATH=.venv/bin/activate

# Activate the virtual environment
source $VENV_PATH


srun python cnn_sweep.py