#! /bin/bash

python ~/phnological-reinflection-pytorch/train_inflection_model.py ~/phonological-reinflection-pytorch/data/finnish-train-high ~/phonological-reinflection-pytorch/data/finnish-dev finnish text 300 100 .09 1.5 --gpu
