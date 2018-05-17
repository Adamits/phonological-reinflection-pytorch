#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

for l in english;
    do
	python "$ROOT"/train_inflection_model_features.py "$ROOT"/data/"$l"-train-high "$ROOT"/data/"$l"-dev "$l" high 50 .08 1.5
done;
