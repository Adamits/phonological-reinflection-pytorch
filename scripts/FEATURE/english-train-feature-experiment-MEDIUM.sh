#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

python "$ROOT"/train_inflection_model_features.py "$ROOT"/data/english-train-medium "$ROOT"/data/english-dev english medium 30 .08 1.5
