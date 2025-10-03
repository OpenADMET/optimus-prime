#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=xgboost
#SBATCH --gres=gpu:1

# Source bashrc
source ${HOME}/.bashrc

# Activate environment
conda activate openadmet_models

# Launch script
openadmet anvil --recipe-path catboost.yaml
