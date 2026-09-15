#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=chemprop_log2fc
#SBATCH --gres=gpu:1

# Source bashrc
source ${HOME}/.bashrc

# Activate environment
conda activate openadmet_models

# Launch script; the output directory is pinned because the stage 2 recipe
# refers to this model by path
openadmet anvil --recipe-path chemprop_log2fc.yaml --output-dir chemprop_log2fc_model
