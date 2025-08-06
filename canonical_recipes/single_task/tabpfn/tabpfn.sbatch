#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=tabpfn
#SBATCH --gres=gpu:1

# Source bashrc
source ${HOME}/.bashrc

# Activate environment
conda activate openadmet4

# Launch script
openadmet anvil --recipe-path tabpfn.yaml
