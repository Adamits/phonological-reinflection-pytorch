#! /bin/bash

python train_inflection_model.py ~/phonological-reinflection-pytorch/data/german-train-high ~/phonological-reinflection-pytorch/data/german-dev german text 300 100 .08 2 --gpu
