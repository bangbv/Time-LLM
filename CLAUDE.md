# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Time-LLM is a reprogramming framework (ICLR 2024) that adapts frozen LLMs (LLaMA-7B, GPT-2, BERT) for time series forecasting. It converts time series patches into text-compatible representations via cross-attention reprogramming, augmented by declarative prompts with statistical context.

Paper: https://arxiv.org/abs/2310.01728

## Setup

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate timellm

# Or pip
pip install -r requirements.txt

# Verify library versions
python utils/check_lib_version.py
```

Key dependencies: pytorch==2.2.2, transformers==4.31.0, accelerate==0.28.0, deepspeed==0.14.0, peft==0.17.1

## Running Experiments

```bash
# Run predefined experiment scripts
bash ./scripts/TimeLLM_ETTh1.sh

# Manual training (single or multi-GPU)
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 8 run_main.py \
  --task_name long_term_forecast \
  --model TimeLLM \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --llm_model LLAMA \
  --llm_dim 4096 \
  --llm_layers 32 \
  --seq_len 512 --label_len 48 --pred_len 96

# M4 benchmark
python run_m4.py [args]

# Pretraining pipeline
python run_pretrain.py [args]
```

## Dataset Setup

Download datasets and place in `./dataset/`. Required structure:
```
dataset/
├── ETT-small/    # ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
├── ECL/
├── Traffic/
├── Weather/
├── m4/
└── prompt_bank/  # Domain descriptions (ETT.txt, Weather.txt, etc.)
```

## Architecture

### Core Model (`models/TimeLLM.py`)

The forward pass in `Model.forecast()`:
1. **Normalize** input time series
2. **Patch Embedding** (`layers/Embed.py:PatchEmbedding`) — splits series into overlapping patches, projects to `d_model`
3. **Prompt Generation** — builds natural language prompts with statistics (min/max, median, trend, FFT-based top-5 lags) prepended with domain description from `dataset/prompt_bank/`
4. **ReprogrammingLayer** (cross-attention) — aligns patch embeddings to LLM vocabulary space using word embeddings as keys/values
5. **Frozen LLM** (LLaMA/GPT-2/BERT) — processes concatenated prompt+data tokens
6. **FlattenHead** — projects LLM output back to time series prediction

Only patch embedding, reprogramming layer, and output head are trained. LLM weights are frozen.

### Key Parameters

| Flag | Options | Description |
|------|---------|-------------|
| `--llm_model` | LLAMA, GPT2, BERT | Backbone LLM |
| `--llm_dim` | 4096 (LLaMA), 768 (GPT-2/BERT) | LLM hidden dim |
| `--task_name` | long_term_forecast, short_term_forecast, imputation, anomaly_detection, classification | Task type |
| `--features` | M, S, MS | Multivariate/univariate modes |
| `--patch_len`, `--stride` | int | Patch embedding parameters |

### Distributed Training

Uses `accelerate` + DeepSpeed ZeRO-2 (`ds_config_zero2.json`) with BF16 precision. The config auto-sizes batch size and gradient accumulation per GPU.

### Data Pipeline

`data_provider/data_factory.py` maps dataset names to loader classes in `data_provider/data_loader.py`. The `data_provider_pretrain/` mirror is used for pretraining only.

### Evaluation

Results are written to `./results/`. Metrics (MAE, MSE, RMSE, MAPE, MSPE) are in `utils/metrics.py`. No automated test suite — evaluation is done via the training scripts with fixed seed 2021.
