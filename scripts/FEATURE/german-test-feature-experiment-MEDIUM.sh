#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

python "$ROOT"/train_inflection_model_features.py "$ROOT"/data/german-train-medium "$ROOT"/data/german-dev german medium 30 .08 1.5
