#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

i=2; while [ $i -le 3 ]; do
	 for s in low medium high; do
	     for l in english german hindi hungarian persian polish russian spanish;\
		 do
		     echo "$l"-"$s"-"$i"
		     python "$ROOT"/train_inflection_model.py "$ROOT"/data/"$l"-train-"$s" "$ROOT"/data/"$l"-dev "$l"-"$i" "$s" text 50 20 .08 1.5 --gpu
	     done;
	 done;
	 i=$(expr $i + 1)
done;

