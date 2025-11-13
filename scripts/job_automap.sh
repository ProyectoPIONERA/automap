#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:a100
#SBATCH --job-name=env_installation
#SBATCH --mem=40G
##------------------------ End job description ------------------------

# set -euo pipefail
# umask 077

# Load huggingface and wandb tokens
source ~/.secrets/hf_wandb.env

log_message() {
	local msg="$1"
	echo "[$(date +"%Y-%M-%D %H-%M-%S")]: $msg"
}

echo "===================================================================="

log_message "PURGE"
module purge

log_message "LOAD CONDA"
module load Anaconda3

log_message "LOAD CUDA: CUDA/12.2.0 cuDNN/8.9.2.26-CUDA-12.2.0"
module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0

log_message "LOAD JAVA: Java/11.0.2"
module load Java/11.0.2
srun java --version

nvidia-smi
PPATH="/home/x258/x258750/workspace/automap"
CONDA="/media/apps/avx512-2021/software/Anaconda3/2025.06-1/bin/conda"
ENV_PREFIX="$PPATH/.venv"
PYTHON="$CONDA run -p $ENV_PREFIX --no-capture-output python"

log_message "Python version:"
srun $PYTHON --version

srun bash "/home/x258/x258750/workspace/automap/datasets/blinkg/bin/llm_basic.sh" --no-backup

unset HUGGINGFACE_HUB_TOKEN HF_TOKEN WANDB_API_KEY
