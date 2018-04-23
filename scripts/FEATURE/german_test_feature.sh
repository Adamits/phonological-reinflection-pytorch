#! /bin/bash

ROOT=~/phonological-reinflection-pytorch
l=german

python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds "$ROOT"/models/encoder-"$l"-feature "$ROOT"/models/decoder-"$l"-feature "$ROOT"/models/char2i-"$l"-feature.pkl "$l" medium feature 1 --symbols2i "$ROOT"/models/symbols2i-"$l"-feature.pkl 

python "$ROOT"/evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --guess "$ROOT"/preds/"$l"-feature-preds --lang "$l" --phone
