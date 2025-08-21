#!/bin/bash

# SBATCH --job-name=vimnet_convnext_small_3e4_bs32x8_ep200
# SBATCH -p preempt
# SBATCH --nodes=1
# SBATCH --cpus-per-task=110
# SBATCH -A marlowe-m000071
# SBATCH -G 8
#SBATCH --error=log/vimnet_convnext_small_3e4_bs32x8_ep200.err
#SBATCH --output=log/vimnet_convnext_small_3e4_bs32x8_ep200.out

# 加载必要的模块
module load slurm
# module load cudatoolkit
# module load cudnn/cuda12/9.3.0.75

# 激活conda环境
source /scratch/m000071/luoxd96/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-gpu

# 检查GPU状态
nvidia-smi

which python

bash train_multiple_distributed.sh