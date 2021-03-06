#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in hindi; do
    python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds "$ROOT"/models/medium/encoder-"$l"-feature "$ROOT"/models/medium/decoder-"$l"-feature  "$ROOT"/models/medium/char2i-"$l"-feature.pkl "$l" medium feature 1 --symbols2i "$ROOT"/models/medium/symbols2i-"$l"-feature.pkl
done;

for l in hindi; do
    python "$ROOT"/evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --guess "$ROOT"/preds/"$l"-feature-preds-medium --lang "$l" --phone --print_disagreement
done;
