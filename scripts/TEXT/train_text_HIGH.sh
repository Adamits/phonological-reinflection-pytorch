#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish;
do
    
    python "$ROOT"/train_inflection_model.py "$ROOT"/data/"$l"-train-high "$ROOT"/data/"$l"-dev "$l" high text 100 20 .08 1.5 --gpu 
done;
