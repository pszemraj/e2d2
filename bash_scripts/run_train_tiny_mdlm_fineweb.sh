#!/bin/bash

# Tiny MDLM training on FineWeb-Edu streaming dataset
# For testing streaming datasets on 8GB GPU

cd /home/pszemraj/workspace/LLM/experimental-arch/diffusion-lm/e2d2

export PYTHONPATH=$(pwd)
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME="${PWD}/.hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=0

# Model architecture - tiny for 8GB GPU
HIDDEN_SIZE=128
INTERMEDIATE_SIZE=384
N_LAYERS=4
SEQ_LENGTH=256

# Hyperparameters
LR=3e-4
WARMUP_DURATION="100ba"
BATCH_SIZE=8
MICRO_BATCH_SIZE=1
MAX_DURATION="1000ba"  # ~1000 batches for testing

# Dataset configuration
DATASET_PATH="HuggingFaceFW/fineweb-edu-score-2"
CONFIG_NAME="CC-MAIN-2024-10"  # Recent snapshot
MAX_SAMPLES=10000  # Limit samples for testing

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base

TAG="tiny_mdlm_fineweb"
RUN_NAME=fineweb_tiny_lr${LR}_bsz${BATCH_SIZE}_${TAG}

NUM_WORKERS=0  # Must be 0 for IterableDataset

/home/pszemraj/miniforge3/envs/e2d2-env/bin/python scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=fineweb_edu_train \
  train_dataset.config_name=${CONFIG_NAME} \
  train_dataset.max_samples=${MAX_SAMPLES} \
  ~dataset@eval_dataset \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="500ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=constant_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  model=mdlm \
  training.compile_backbone=false \
  model.config.length=${SEQ_LENGTH} \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.backbone_config.reinit_model=true \
  model.config.backbone_config.num_layers=${N_LAYERS} \
  model.config.backbone_config.keep_top_layers=false \
  +model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  +model.config.backbone_config.intermediate_size=${INTERMEDIATE_SIZE} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / MICRO_BATCH_SIZE )) \
  training.antithetic_sampling=true \
  hydra.run.dir=outputs/${RUN_NAME} \
  composer.trainer.save_interval="500ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.save_to_hub=false \
  composer.trainer.autoresume=false \
  hydra.job.chdir=false
