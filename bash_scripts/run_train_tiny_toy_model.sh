#!/bin/bash

# Minimal training script for 8GB GPU - tiny toy model
# This is a test setup to verify the training pipeline works

# Setup environment
cd "$(dirname "$0")/.." || exit  # Go to the root directory of the repo
source setup_env.sh

# Tiny model architecture for 8GB GPU
HIDDEN_SIZE=128
INTERMEDIATE_SIZE=384
N_LAYERS=4
SEQ_LENGTH=256

# Conservative hyperparameters
LR=3e-4
WARMUP_DURATION="50ba"
BATCH_SIZE=8
MAX_DURATION="500ba"  # Short run to test

# Use GPT2 as base (smallest available pretrained model)
PRETRAINED_MODEL_NAME_OR_PATH=openai-community/gpt2

TAG="tiny_toy"
LAYERS="layers${N_LAYERS}"
RUN_NAME=tiny_${TAG}_lr${LR}_bsz${BATCH_SIZE}_${LAYERS}_hidden${HIDDEN_SIZE}_seq${SEQ_LENGTH}

# For 8GB GPU, use very small micro batch size
MICRO_BATCH_SIZE=1
NUM_WORKERS=2
NUM_VISIBLE_DEVICES=${NUM_VISIBLE_DEVICES:-1}

echo "======================================"
echo "Training tiny toy model for 8GB GPU"
echo "Model: MDLM with ${N_LAYERS} layers"
echo "Hidden size: ${HIDDEN_SIZE}"
echo "Sequence length: ${SEQ_LENGTH}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max duration: ${MAX_DURATION}"
echo "======================================"

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=cnn_dailymail_train \
  dataset@eval_dataset=cnn_dailymail_eval \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="100ba" \
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
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  training.antithetic_sampling=false \
  composer.trainer.save_interval="100ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true
