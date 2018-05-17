#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

i=3; while [ $i -le 3 ]; do
	 for s in low medium high; do
	     for l in english german hindi hungarian persian polish russian spanish;
	     do
		 python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds "$ROOT"/models/"$s"/encoder-"$l"-"$i"-text "$ROOT"/models/"$s"/decoder-"$l"-"$i"-text "$ROOT"/models/"$s"/char2i-"$l"-"$i"-text.pkl "$l"-"$i" "$s" text 1 --gpu
	     done;
	 done;
done;
