#!/bin/bash

## Resource allocation for Mamba model on QQP classification
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gres=shard:2
#SBATCH --qos=gpu

## Job info
#SBATCH --job-name=qqp_mamba
#SBATCH --output=qqp_mamba-%j.out
#SBATCH --error=qqp_mamba-%j.err

## Time limit (48 hours)
#SBATCH --time=48:00:00

## Initialize and activate conda environment
source ~/.bashrc
conda activate lmu_s4

## Navigate to project directory
cd /RG/rg-tsur/bagal58/LMU_S4

## Run training
python src/notebooks/qqp/training/train_qqp_mamba.py \
    --device cuda \
    --amp \
    --evaluate \
    --num_workers 8 \
    --batch 256 \
    --epochs 100

echo "Training completed at $(date)"

