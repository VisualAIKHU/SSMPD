#!/bin/bash
# Evaluate the released model on the KAIST test set.
cd "$(dirname "$0")/src" || exit 1
MODEL=${1:-../weights/SSMPD_KAIST_10p.pth.tar}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
python inference.py --FDZ original --model-path "$MODEL" --result-dir ../result
