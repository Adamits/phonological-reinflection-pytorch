#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

python "$ROOT"/train_inflection_model.py "$ROOT"/data/german-train-medium "$ROOT"/data/german-dev german mediumphone 30 1 .08 1.5
