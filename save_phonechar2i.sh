#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish;\
    do
	python "$ROOT"/save_phonechar2i_dicts.py "$ROOT"/data/"$l"-train-low "$l"
done;
