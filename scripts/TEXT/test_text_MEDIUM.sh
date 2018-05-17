#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds "$ROOT"/models/medium/encoder-"$l"-text "$ROOT"/models/medium/decoder-"$l"-text "$ROOT"/models/medium/char2i-"$l"-text.pkl "$l" medium text 1
done;

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --guess "$ROOT"/preds/"$l"-text-preds-medium --lang "$l"
done;
