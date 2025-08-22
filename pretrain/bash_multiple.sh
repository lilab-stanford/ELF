#!/bin/bash

# SBATCH --job-name=vimnet_convnext_small_3e4_bs32x8_ep200
# SBATCH -p preempt
# SBATCH --nodes=1
# SBATCH --cpus-per-task=110
# SBATCH -A marlowe-m000071
# SBATCH -G 8
#SBATCH --error=log/vimnet_convnext_small_3e4_bs32x8_ep200.err
#SBATCH --output=log/vimnet_convnext_small_3e4_bs32x8_ep200.out

module load slurm
# module load cudatoolkit
# module load cudnn/cuda12/9.3.0.75

source /scratch/m000071/luoxd96/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-gpu

nvidia-smi

which python

bash train_multiple_distributed.sh
