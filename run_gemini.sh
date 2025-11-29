#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --partition=gpu4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=UBAIJOB 
#SBATCH -o ./logs/jupyter.%N.%j.out  # STDOUT 
#SBATCH -e ./logs/jupyter.%N.%j.err  # STDERR

echo "start at:" `date` 
echo "node: $HOSTNAME" 
echo "jobid: $SLURM_JOB_ID" 

module unload CUDA/11.2.2 
module load cuda/11.8.0

export PYTHONPATH=$PYTHONPATH:.

echo "letsgo"

python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('Current CUDA device:', torch.cuda.current_device()); print('CUDA device name:', torch.cuda.get_device_name(torch.cuda.current_device()))"
python -u gemini_api/run.py