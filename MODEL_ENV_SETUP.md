# Fast-dLLM v2 环境配置记录（eic_onboarding / eic_onboarding_math310）

本文件只记录模型最开始的环境配置内容，方便其他人直接复现。

## 先看这个提示（重要）

建议把 conda 环境和缓存都放在 `scratch`，不要放在 `home`。  
原因：模型相关依赖和缓存体积大，放在 `scratch` 可以避免明显占用 `home` 空间。

推荐先执行：

```bash
mkdir -p /home/hice1/hhao40/scratch/conda_envs
mkdir -p /home/hice1/hhao40/scratch/conda_pkgs
mkdir -p /home/hice1/hhao40/scratch/hf_home
mkdir -p /home/hice1/hhao40/scratch/pip_cache

conda config --add envs_dirs /home/hice1/hhao40/scratch/conda_envs
conda config --add pkgs_dirs /home/hice1/hhao40/scratch/conda_pkgs
export PIP_CACHE_DIR=/home/hice1/hhao40/scratch/pip_cache
```

## 两个环境分别是什么

1. `eic_onboarding`
用于 Fast-dLLM v2 的主环境（我跑 GSM8K 时主要用它），Python 3.9，包含 `mpi4py`、`torch`、`transformers`、`accelerate` 等依赖。

2. `eic_onboarding_math310`
用于 MATH 相关评测环境（我跑 `minerva_math` 时主要用它），Python 3.10，额外确认了 `lm-eval[math]==0.4.8`。

必须单独建 `eic_onboarding_math310 (python=3.10)` 的主要原因：  
需要 `math_verify`（由 `lm-eval[math]` 安装）来做数学答案等价校验。  
这个校验不是只看字符串是否完全一致，而是“表达式/结论语义等价也判对”。

当前环境路径（`conda info --envs`）：

- `/home/hice1/hhao40/scratch/conda_envs/eic_onboarding`
- `/home/hice1/hhao40/scratch/conda_envs/eic_onboarding_math310`

当前已验证的差异（重点）：

- `eic_onboarding_math310`：有 `math-verify 0.9.0`，`lm-eval 0.4.8`
- `eic_onboarding`：有 `lm-eval 0.4.8`，但没有 `math-verify`

## 我当时的实际创建过程（按历史记录）

来自 `conda-meta/history` 与 shell 历史：

1. 创建 `eic_onboarding`

```bash
conda create -n eic_onboarding python=3.9 -y
conda install -y mpi4py
```

2. 创建 `eic_onboarding_math310`

```bash
conda create -p /home/hice1/hhao40/scratch/conda_envs/eic_onboarding_math310 python=3.10 pip -y
```

3. 在 `eic_onboarding_math310` 中重新安装数学评测依赖

```bash
/home/hice1/hhao40/scratch/conda_envs/eic_onboarding_math310/bin/pip uninstall -y lm-eval
/home/hice1/hhao40/scratch/conda_envs/eic_onboarding_math310/bin/pip install "lm-eval[math]==0.4.8"
```

## 可复现的配置步骤（从零开始）

下面是按当前可用环境快照整理的可复现命令。

### A) 复现 `eic_onboarding`（Python 3.9）

```bash
conda create -p /home/hice1/hhao40/scratch/conda_envs/eic_onboarding python=3.9 -y
conda activate /home/hice1/hhao40/scratch/conda_envs/eic_onboarding
conda install -y mpi4py

pip install \
  torch==2.8.0 transformers==4.53.1 tokenizers==0.21.4 safetensors==0.7.0 sentencepiece==0.2.1 \
  huggingface-hub==0.36.2 accelerate==0.34.2 datasets==4.5.0 evaluate==0.4.0 \
  numpy==2.0.2 scipy==1.13.1 pandas==2.3.3 \
  lm-eval==0.4.8 einops==0.8.2 tqdm==4.67.3 triton==3.4.0 peft==0.17.1 \
  bitsandbytes==0.48.2 deepspeed==0.18.5
```

### B) 复现 `eic_onboarding_math310`（Python 3.10）

```bash
conda create -p /home/hice1/hhao40/scratch/conda_envs/eic_onboarding_math310 python=3.10 pip -y
conda activate /home/hice1/hhao40/scratch/conda_envs/eic_onboarding_math310

pip install \
  torch==2.8.0 transformers==4.53.1 tokenizers==0.21.4 safetensors==0.7.0 sentencepiece==0.2.1 \
  huggingface-hub==0.36.2 accelerate==1.13.0 datasets==4.8.4 evaluate==0.4.0 \
  numpy==2.2.6 scipy==1.15.3 pandas==2.3.3 \
  einops==0.8.2 tqdm==4.67.3 triton==3.4.0 peft==0.18.1

pip install "lm-eval[math]==0.4.8"
```

安装后可确认：

```bash
python -c "from importlib.metadata import version; print('lm-eval', version('lm-eval')); print('math-verify', version('math-verify'))"
```

## 环境检查（安装后建议执行）

```bash
# 检查 eic_onboarding
conda activate /home/hice1/hhao40/scratch/conda_envs/eic_onboarding
python -V
python -c "from importlib.metadata import version; import torch, transformers, accelerate, datasets; print('torch', torch.__version__); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('datasets', datasets.__version__); print('lm-eval', version('lm-eval'))"

# 检查 eic_onboarding_math310
conda activate /home/hice1/hhao40/scratch/conda_envs/eic_onboarding_math310
python -V
python -c "from importlib.metadata import version; import torch, transformers, accelerate, datasets; print('torch', torch.__version__); print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('datasets', datasets.__version__); print('lm-eval', version('lm-eval')); print('math-verify', version('math-verify'))"
```

说明：当前两套环境中的 `torch` 都是 `2.8.0+cu128`（CUDA 12.8 构建）。
