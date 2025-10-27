#!/bin/bash

# Tiny MDLM training for 8GB GPU testing
# Based on run_train_mdlm_cnn.sh with reduced parameters

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

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B-Base

TAG="tiny_mdlm"
RUN_NAME=cnn_tiny_lr${LR}_bsz${BATCH_SIZE}_${TAG}

NUM_WORKERS=2

/home/pszemraj/miniforge3/envs/e2d2-env/bin/python scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=cnn_dailymail_train \
  dataset@eval_dataset=cnn_dailymail_eval \
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
