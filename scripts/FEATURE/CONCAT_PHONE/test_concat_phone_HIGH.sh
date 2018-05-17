#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds "$ROOT"/models/high/encoder-"$l"-feature_concat "$ROOT"/models/high/decoder-"$l"-feature_concat "$ROOT"/models/high/char2i-"$l"-feature_concat.pkl "$l" high feature 1 --symbols2i "$ROOT"/models/high/symbols2i-"$l"-feature_concat.pkl --phonechar2i "$ROOT"/models/high/phonechar2i-"$l"-feature_concat.pkl --concat_phone
done;

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --guess "$ROOT"/preds/"$l"-feature_concat-preds-high --lang "$l" --phone
done;
