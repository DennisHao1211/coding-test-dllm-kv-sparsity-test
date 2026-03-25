#!/bin/bash
#SBATCH -J math_base_1p5b
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /home/hice1/hhao40/Fast-dLLM/v2/slurm_math/%x-%j.out

set -euo pipefail

source /home/hice1/hhao40/scratch/miniconda3/etc/profile.d/conda.sh
conda activate /home/hice1/hhao40/scratch/conda_envs/eic_onboarding_math310

cd /home/hice1/hhao40/Fast-dLLM/v2

accelerate launch eval.py \
  --tasks minerva_math \
  --batch_size 1 \
  --num_fewshot 0 \
  --confirm_run_unsafe_code \
  --model fast_dllm_v2 \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --model_args "model_path=/home/hice1/hhao40/Fast-dLLM/v2/local_fast_dllm_1.5B,code_path=/home/hice1/hhao40/Fast-dLLM/v2/modified,local_files_only=True,use_block_cache=True,use_prefix_filter=False,bd_size=32,small_block_size=16,threshold=0.9"
