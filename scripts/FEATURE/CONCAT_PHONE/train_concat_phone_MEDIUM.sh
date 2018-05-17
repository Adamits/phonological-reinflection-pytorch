#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish;\
    do
	echo "$l"
	python "$ROOT"/train_inflection_model_features.py "$ROOT"/data/"$l"-train-medium "$ROOT"/data/"$l"-dev "$l" medium 50 .08 1.5 --concat_phone
done;
