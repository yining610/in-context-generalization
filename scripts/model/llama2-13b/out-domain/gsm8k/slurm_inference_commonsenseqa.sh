#!/bin/bash

#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=60:00:00
#SBATCH --job-name="llama2-13b-ood-csqa-gsm8k-f-m4096"
#SBATCH --output="./screen_log/llama2-"
#SBATCH --mem=20G
#SBATCH --mail-user=ylu130@jh.edu
#SBATCH --mail-type=ALL

