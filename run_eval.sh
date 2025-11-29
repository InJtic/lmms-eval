#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --partition=gpu2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --job-name=ddong_eval
#SBATCH -w n052
#SBATCH -o ./logs/jupyter.%N.%j.out  # STDOUT 
#SBATCH -e ./logs/jupyter.%N.%j.err  # STDERR

echo "start at:" `date` 

module unload CUDA/11.2.2 
module load cuda/11.8.0

export PYTHONPATH=$PYTHONPATH:./lmms_eval
source .env

MODEL_TYPE="internvl2"
MODEL_ARGS="pretrained=OpenGVLab/InternVL3_5-8B"
# MODEL_TYPE="qwen3_vl"
# MODEL_ARGS="pretrained=Qwen/Qwen3-VL-8B-Instruct"
# MODEL_TYPE="llava_vid"
# MODEL_ARGS="pretrained=lmms-lab/LLaVA-Video-7B-Qwen2"
# MODEL_TYPE="phi4_multimodal" 
# MODEL_ARGS="pretrained=Lexius/Phi-4-multimodal-instruct"

echo "node: $HOSTNAME" 
echo "jobid: $SLURM_JOB_ID, model_type: $MODEL_TYPE"

OUTPUT_DIR="./output/results"

python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('Current CUDA device:', torch.cuda.current_device()); print('CUDA device name:', torch.cuda.get_device_name(torch.cuda.current_device()))"

accelerate launch --num_processes 4 \
   -m lmms_eval \
   --model $MODEL_TYPE \
   --model_args $MODEL_ARGS \
   --tasks ddong_black_vs_noise \
   --batch_size 1 \
   --log_samples \
   --output_path $OUTPUT_DIR

accelerate launch --num_processes 4 \
   -m lmms_eval \
   --model $MODEL_TYPE \
   --model_args $MODEL_ARGS \
   --tasks ddong_center_vs_random \
   --batch_size 1 \
   --log_samples \
   --output_path $OUTPUT_DIR

accelerate launch --num_processes 4 \
   -m lmms_eval \
   --model $MODEL_TYPE \
   --model_args $MODEL_ARGS \
   --tasks ddong_color_vs_wb \
   --batch_size 1 \
   --log_samples \
   --output_path $OUTPUT_DIR

accelerate launch --num_processes 4 \
   -m lmms_eval \
   --model $MODEL_TYPE \
   --model_args $MODEL_ARGS \
   --tasks ddong_direction \
   --batch_size 1 \
   --log_samples \
   --output_path $OUTPUT_DIR
