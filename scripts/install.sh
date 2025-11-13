#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --ntasks=1
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:v100
#SBATCH --job-name=env_installation
#SBATCH --mem-per-cpu=8000
##------------------------ End job description ------------------------

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
	
nvidia-smi
PPATH="/home/x258/x258750/workspace/automap"
CONDA="/media/apps/avx512-2021/software/Anaconda3/2025.06-1/bin/conda"
ENV_PREFIX="$PPATH/.venv"
PYTHON="$CONDA run -p $ENV_PREFIX --no-capture-output python"

log_message "Python version:"
srun $PYTHON --version

log_message "Instalando env.yml"
# srun $CONDA env update -p "$ENV_PREFIX" --file="$PPATH/environment.yml" 
# srun $CONDA install -p "$ENV_PREFIX" -y -c pytorch -c nvidia \
#	pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.1

log_message "Probando pytorch"
srun $PYTHON "$PPATH/scripts/test_torch.py"
