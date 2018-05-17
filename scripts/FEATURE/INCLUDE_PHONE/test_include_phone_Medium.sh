#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds "$ROOT"/models/medium/encoder-"$l"-feature_phone "$ROOT"/models/medium/decoder-"$l"-feature_phone "$ROOT"/models/medium/char2i-"$l"-feature_phone.pkl "$l" medium feature 1 --symbols2i "$ROOT"/models/medium/symbols2i-"$l"-feature_phone.pkl --include_phone
done;

for l in english german hindi hungarian persian polish russian spanish; do
    python "$ROOT"/evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --guess "$ROOT"/preds/"$l"-feature_phone-preds-medium --lang "$l" --phone
done;
