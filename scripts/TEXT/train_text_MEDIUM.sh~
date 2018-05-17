#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish;\
    do
	echo "$l"
	python "$ROOT"/train_inflection_model.py "$ROOT"/data/"$l"-train-low "$ROOT"/data/"$l"-dev "$l" low text 50 1 .08 1.5
done;
