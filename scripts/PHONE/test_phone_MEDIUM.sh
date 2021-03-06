#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds "$ROOT"/models/medium/encoder-"$l"-phone "$ROOT"/models/medium/decoder-"$l"-phone "$ROOT"/models/medium/char2i-"$l"-phone.pkl "$l" medium phone 1 --gpu
done;

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --guess "$ROOT"/preds/"$l"-phone-preds-medium --lang "$l" --phone
done;
