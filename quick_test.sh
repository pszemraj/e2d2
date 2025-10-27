#!/bin/bash
# Quick diagnostic test - shows full errors

cd /home/pszemraj/workspace/LLM/experimental-arch/diffusion-lm/e2d2

export PYTHONPATH=$(pwd)
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1
export HF_HOME="${PWD}/.hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=0

/home/pszemraj/miniforge3/envs/e2d2-env/bin/python scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=quick_test \
  pretrained_model_name_or_path=Qwen/Qwen3-0.6B-Base \
  dataset@train_dataset=cnn_dailymail_train \
  dataset@eval_dataset=cnn_dailymail_eval \
  model=mdlm \
  model.config.length=256 \
  model/backbone@model.config.backbone_config=automodel_for_causal_lm \
  model.config.backbone_config.reinit_model=true \
  model.config.backbone_config.num_layers=4 \
  +model.config.backbone_config.hidden_size=128 \
  +model.config.backbone_config.intermediate_size=384 \
  training.global_batch_size=2 \
  training.grad_accum=2 \
  training.autoresume=false \
  composer.trainer.max_duration=10ba \
  composer.trainer.autoresume=false \
  training.compile_backbone=false \
  train_dataloader.num_workers=2 \
  composer.callbacks.hf_compatible_checkpointing.save_to_hub=false \
  hydra.job.chdir=false
