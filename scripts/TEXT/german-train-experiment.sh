#! /bin/bash

python ~/phonological-reinflection-pytorch/train_inflection_model.py ~/phonological-reinflection-pytorch/data/german-train-high ~/phonological-reinflection-pytorch/data/german-dev german high text 300 100 .08 2 --gpu
