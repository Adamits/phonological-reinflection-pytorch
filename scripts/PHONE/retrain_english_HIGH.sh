#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

for l in english;
    do
	echo "$l"
	python "$ROOT"/train_inflection_model.py "$ROOT"/data/"$l"-train-high "$ROOT"/data/"$l"-dev "$l" high phone 50 1 .08 1.5
done;
