#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish;\
    do
	echo "$l"
	python "$ROOT"/train_inflection_model_features.py "$ROOT"/data/"$l"-train-low "$ROOT"/data/"$l"-dev "$l" low 150 .08 1.5 --concat_phone
done;
