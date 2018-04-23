#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in bengali catalan dutch english french german hindi hungarian italian kurmanji persian polish portuguese russian sorani spanish swedish ukrainian; do
    python "$ROOT"/train_inflection_model.py "$ROOT"/data/"$l"-train-high\
	   "$ROOT"/data/"$l"-dev "$l" high phone 100 20 .08 1.5 --gpu 
done;
