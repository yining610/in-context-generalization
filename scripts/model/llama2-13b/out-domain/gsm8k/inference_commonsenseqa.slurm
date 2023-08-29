#!/bin/bash
#SBATCH -A danielk_gpu
#SBATCH --partition a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=60:00:00
#SBATCH --job-name="llama2-13b-ood-csqa-gsm8k-f-m4096"
#SBATCH --output="./screen_log/llama2-13b/csqa_gsm8k_ood1-25_f_4096.txt"
#SBATCH --mem=32G
#SBATCH --mail-user=ylu130@jh.edu
#SBATCH --mail-type=ALL

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge
module load anaconda

conda activate ood # open the Python environment

bash 


