#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds "$ROOT"/models/high/encoder-"$l"-text "$ROOT"/models/high/decoder-"$l"-text "$ROOT"/models/high/char2i-"$l"-text.pkl "$l" high text 1 --gpu
done;

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --guess "$ROOT"/preds/"$l"-text-preds-high --lang "$l"
done;
