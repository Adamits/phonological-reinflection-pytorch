#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds "$ROOT"/models/low/encoder-"$l"-phone "$ROOT"/models/low/decoder-"$l"-phone "$ROOT"/models/low/char2i-"$l"-phone.pkl "$l" low phone 1
done;

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --guess "$ROOT"/preds/"$l"-phone-preds-low --lang "$l" --phone
done;
