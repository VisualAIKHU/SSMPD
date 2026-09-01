#!/bin/bash
# Train SSMPD. Set dataset_type / percentage / checkpoints in src/config.py first.
cd "$(dirname "$0")/src" || exit 1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} python train.py
