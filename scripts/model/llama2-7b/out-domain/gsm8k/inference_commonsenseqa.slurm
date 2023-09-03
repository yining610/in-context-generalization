#!/bin/bash
#SBATCH -A danielk_gpu
#SBATCH --partition ica100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=36:00:00
#SBATCH --job-name="ood11-20-csqa-gsm8k-t-m4096"
#SBATCH --output="./screen_log/llama2-7b/csqa_gsm8k_ood11-20_t_4096.txt"
#SBATCH --mem=80G
#SBATCH --array=0-1 # job array index
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

cd /home/ylu130/workspace/in-context-generalization/scripts/model/llama2-7b/out-domain/gsm8k

bash ./inference_commonsenseqa.sh "$(($SLURM_ARRAY_TASK_ID*5+11)) $(($SLURM_ARRAY_TASK_ID*5+12)) $(($SLURM_ARRAY_TASK_ID*5+13)) $(($SLURM_ARRAY_TASK_ID*5+14)) $(($SLURM_ARRAY_TASK_ID*5+15))" "True" "4096" "10"
