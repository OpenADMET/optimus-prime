#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=tabicl_pec50
#SBATCH --gres=gpu:1

# Source bashrc
source ${HOME}/.bashrc

# Activate environment
conda activate openadmet_models

# Launch script; requires the stage 1 model to already exist at
# ../chemprop_log2fc/chemprop_log2fc_model
openadmet anvil --recipe-path tabicl_pec50.yaml
