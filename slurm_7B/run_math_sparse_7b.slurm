#!/bin/bash
#SBATCH -J math_sparse_7b
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /home/hice1/hhao40/Fast-dLLM/v2/slurm_7b/%x-%j.out

set -euo pipefail

source /home/hice1/hhao40/scratch/miniconda3/etc/profile.d/conda.sh
conda activate /home/hice1/hhao40/scratch/conda_envs/eic_onboarding_math310

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME=/home/hice1/hhao40/scratch/hf_home
export TRANSFORMERS_CACHE=/home/hice1/hhao40/scratch/hf_home
export HF_HUB_DISABLE_XET=1
mkdir -p /home/hice1/hhao40/scratch/hf_home

cd /home/hice1/hhao40/Fast-dLLM/v2

accelerate launch eval.py   --tasks minerva_math   --batch_size 1   --num_fewshot 0   --confirm_run_unsafe_code   --model fast_dllm_v2   --fewshot_as_multiturn   --apply_chat_template   --model_args "model_path=Efficient-Large-Model/Fast_dLLM_v2_7B,code_path=/home/hice1/hhao40/Fast-dLLM/v2/modified,use_block_cache=True,use_prefix_filter=True,bd_size=32,small_block_size=16,threshold=0.9,block_cache_keep_ratio=0.7,delay_step=1,block_cache_pool_kernel_size=3"
