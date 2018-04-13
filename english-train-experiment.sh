#! /bin/bash

python train_inflection_model.py ~/phonological-reinflection-pytorch/data/english-train-high ~/phonological-reinflection-pytorch/data/english-dev english text 200 100 .08 1.75 --gpu
