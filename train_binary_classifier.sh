#!/usr/bin/env bash

source /home/young/ws_aic/aic_hilserl_env/pixi_env_setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}/examples"

GPU_UUID="${GPU_UUID:-GPU-45a924c1-5d32-a7cc-ed8d-674d0179dcee}"
EXP_NAME="${EXP_NAME:-aic_cable_insertion}"
NUM_EPOCHS="${NUM_EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
PYTHON_BIN="${PYTHON_BIN:-/home/young/ws_aic/aic_hilserl_env/.pixi/envs/default/bin/python3}"
DATASET_DIR="${DATASET_DIR:-/home/young/ws_aic/aic_hilserl_env/classifier_data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/young/ws_aic/aic_hilserl_env/classifier_ckpt}"

echo "Using GPU UUID ${GPU_UUID} for reward classifier training."
echo "exp_name=${EXP_NAME} num_epochs=${NUM_EPOCHS} batch_size=${BATCH_SIZE}"
echo "steps_per_epoch=${STEPS_PER_EPOCH} learning_rate=${LEARNING_RATE}"
echo "dataset_dir=${DATASET_DIR}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"

cd "${EXAMPLES_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_UUID}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

exec "${PYTHON_BIN}" train_reward_classifier.py \
  --exp_name="${EXP_NAME}" \
  --num_epochs="${NUM_EPOCHS}" \
  --batch_size="${BATCH_SIZE}" \
  --steps_per_epoch="${STEPS_PER_EPOCH}" \
  --learning_rate="${LEARNING_RATE}" \
  --dataset_dir="${DATASET_DIR}" \
  --checkpoint_dir="${CHECKPOINT_DIR}" \
  "$@"
