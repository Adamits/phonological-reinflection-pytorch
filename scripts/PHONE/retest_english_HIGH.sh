#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in english; do
    python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds "$ROOT"/models/high/encoder-"$l"-phone "$ROOT"/models/high/decoder-"$l"-phone "$ROOT"/models/high/char2i-"$l"-phone.pkl "$l" high phone 1 --gpu
done;

for l in english; do
    python "$ROOT"/evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --guess "$ROOT"/preds/"$l"-phone-preds-high --lang "$l" --phone --print_disagreement
done;
