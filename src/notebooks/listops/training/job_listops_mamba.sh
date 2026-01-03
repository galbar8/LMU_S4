#!/bin/bash

## Resource allocation for large Mamba model on ListOps
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gres=shard:2
#SBATCH --qos=gpu

## Job info
#SBATCH --job-name=listops_mamba
#SBATCH --output=listops_mamba-%j.out
#SBATCH --error=listops_mamba-%j.err

## Time limit (48 hours for heavy training)
#SBATCH --time=48:00:00

## Initialize and activate conda environment
source ~/.bashrc
conda activate lmu_s4

## Navigate to project directory
cd /RG/rg-tsur/bagal58/LMU_S4

## Run training - uses script defaults for Mamba parameters
python src/notebooks/listops/training/train_listops_mamba.py \
    --device cuda \
    --amp \
    --evaluate \
    --num_workers 8 \
    --batch 128 \
    --max_length 512

echo "Training completed at $(date)"

