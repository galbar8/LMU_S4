#!/bin/bash

## Resource allocation for Mamba model on PPG regression
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --gres=shard:2
#SBATCH --qos=gpu

## Job info
#SBATCH --job-name=ppg_mamba
#SBATCH --output=ppg_mamba-%j.out
#SBATCH --error=ppg_mamba-%j.err

## Time limit (48 hours for heavy training)
#SBATCH --time=48:00:00

## Initialize and activate conda environment
source ~/.bashrc
conda activate lmu_s4

## Navigate to project directory
cd /RG/rg-tsur/bagal58/LMU_S4

## Run training - uses script defaults for Mamba parameters
python src/notebooks/ppg/training/train_ppg_mamba.py \
    --device cuda \
    --amp \
    --evaluate \
    --num_workers 8 \
    --batch 128 \
    --epochs 100

echo "Training completed at $(date)"

