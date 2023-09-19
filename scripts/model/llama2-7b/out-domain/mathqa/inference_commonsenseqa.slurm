#!/bin/bash -l
#SBATCH -A ia1
#SBATCH --partition debug
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name="ood1-20-csqa-mathqa-b-m4096"
#SBATCH --output="./screen_log/llama2-7b/csqa_mathqa_ood1-20_b_4096.txt"
#SBATCH --array=0-4
#SBATCH --mail-user=ylu130@jh.edu
#SBATCH --mail-type=ALL

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
eval "$__conda_setup"
else
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then . "/opt/anaconda3/etc/profile.d/conda.sh"
else
export PATH="/opt/anaconda3/bin:$PATH"
fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate ood

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

cd /home/ylu130/workspace/in-context-generalization/scripts/model/llama2-7b/out-domain/mathqa

bash ./inference_commonsenseqa.sh "$(($SLURM_ARRAY_TASK_ID*4+1)) $(($SLURM_ARRAY_TASK_ID*4+3))" "True False" "4096" "5"
