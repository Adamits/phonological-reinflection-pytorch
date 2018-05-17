#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

i=1; while [ $i -le 3 ]; do
	 for s in low; do
	     for l in english german hindi hungarian persian polish russian spanish; do
		     echo "$l"-"$s"-"$i"
		     python "$ROOT"/train_inflection_model.py "$ROOT"/data/"$l"-train-"$s" "$ROOT"/data/"$l"-dev "$l" "$s" text 50 1 .08 1.5 --i "$i"
	     done;
	 done;
	 i=$(expr $i + 1)
done;

