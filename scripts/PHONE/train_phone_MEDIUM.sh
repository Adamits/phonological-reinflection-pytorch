#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish;\
    do
	python "$ROOT"/train_inflection_model.py "$ROOT"/data/"$l"-train-medium "$ROOT"/data/"$l"-dev "$l" medium phone 50 20 .08 1.5 --gpu
done;
