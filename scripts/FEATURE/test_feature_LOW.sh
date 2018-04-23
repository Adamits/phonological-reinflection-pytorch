#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds\
	   "$ROOT"/models/low/encoder-"$l"-feature "$ROOT"/models/low/decoder-"$l"-feature\
	   "$ROOT"/models/low/char2i-"$l"-feature.pkl "$l" low feature 1 --symbols2i\
	   "$ROOT"/models/low/symbols2i-"$l"-feature.pkl 

python "$ROOT"/evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --guess "$ROOT"/preds/"$l"-feature-preds-low --lang "$l" --phone
