#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in german; do
    python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds "$ROOT"/models/high/encoder-"$l"-feature "$ROOT"/models/high/decoder-"$l"-feature "$ROOT"/models/high/char2i-"$l"-feature.pkl "$l" high feature 1 --gpu
done;

for l in german; do
    python "$ROOT"/evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --guess "$ROOT"/preds/"$l"-feature-preds-high --lang "$l" --phone --print_disagreement
done;
