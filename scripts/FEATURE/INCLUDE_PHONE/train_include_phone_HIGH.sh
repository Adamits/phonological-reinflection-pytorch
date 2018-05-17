#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish;\
    do
	echo "$l"
	python "$ROOT"/train_inflection_model_features.py "$ROOT"/data/"$l"-train-high "$ROOT"/data/"$l"-dev "$l" high 50 .08 1.5 --include_phone
done;
