#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in english german hindi hungarian persian polish russian spanish; do
    echo "$l"
    python "$ROOT"/oracle_evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --text_guess "$ROOT"/preds/"$l"-text-preds-medium --phone_guess "$ROOT"/preds/"$l"-phone-preds-medium --lang "$l" --feature_guess "$ROOT"/preds/"$l"-feature-preds-medium
done;
