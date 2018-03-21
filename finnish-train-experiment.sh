#! /bin/bash

BATCH_SIZE="$1"
EPOCHS="$2"
LR="$3"

python train.py ./data/finnish-train-high ./data/finnish-dev finnish "$BATCH_SIZE" "$EPOCHS" "$LR" --gpu
