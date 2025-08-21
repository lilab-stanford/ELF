#!/bin/bash

# SBATCH --job-name=multiple_model_5fms_8gpus_nf_4096_bs_768
# SBATCH -p preempt
# SBATCH --nodes=1
# SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
# SBATCH -A marlowe-m000071
# SBATCH -G 8
#SBATCH --error=log/multiple_model_5fms_8gpus_nf_4096_bs_768.err
#SBATCH --output=log/multiple_model_5fms_8gpus_nf_4096_bs_768.out

module load slurm
# module load cudatoolkit
# module load cudnn/cuda12/9.3.0.75


conda activate py310

nvidia-smi

which python

# config_file=config/vimnet_convnext_small_3e4_bs32x8_ep200.py

# base_dir=/scratch/m000071/yfj/projects/vim/

# work_dir=$(echo ${config_file%.*} | sed -e "s/config/work_dirs/g")/
# work_dir=${base_dir}/${work_dir}



# accelerate launch --multi_gpu --num_processes=8 tools/train.py $config_file --work-dir $work_dir


# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# srun -p preempt -A marlowe-m000071 --nodes=1 --gpus-per-node=8 --time=12:00:00 --cpus-per-task=110 --pty bash



python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    pretrain_attention_weight_multiple_model.py \
    --exp-name multiple_model_5fms_8gpus_nf_4096_bs_768_no_proj \
    --data-path /scratch/m000071/luoxd96/elf/features \
    --csv-path /scratch/m000071/luoxd96/elf/features/merged_dataset_split_cleaned.csv \
    --output-dir /scratch/m000071/luoxd96/elf/checkpoints/multiple_model_5fms_8gpus_nf_4096_bs_768_no_proj \
    --gpu 0 1 2 3 4 5 6 7 \
    --l-dim 768 \
    --dim 768 \
    --feature-models conch_v1_5 gigapath h0 uni virchow2 \
    --batch-size 768 \
    --dist-url 'env://' \
    --multiprocessing-distributed \
    --world-size 8 \
    --num-feats 4096 \
    --mixed-precision \
    --resume /scratch/m000071/luoxd96/elf/checkpoints/multiple_model_5fms_8gpus_nf_4096_bs_768_no_proj/multiple_model_5fms_8gpus_nf_4096_bs_768_no_proj/checkpoint_latest.pth
    # --resume /scratch/m00/0071/luoxd96/elf/checkpoints/multiple_model_5fms_8gpus_nf_4096_bs_768/multiple_model_5fms_8gpus_nf_4096_bs_768/checkpoint_latest.pth
