#!/usr/bin/bash
#
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --gres=gpu:4
#SBATCH --output=log_train_%A_%a.out
#SBATCH --error=log_train_%A_%a.err
#SBATCH --time=48:00:00

module purge

source $HOME/.bashrc

conda activate ke_gpu

module load cuda/10.1




# train a resnet50 model
python resnet50_copy.py \
  --output_dir ./output >& train_server_version2.log
