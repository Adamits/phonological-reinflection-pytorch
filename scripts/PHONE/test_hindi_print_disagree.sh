#! /bin/bash

ROOT=~/phonological-reinflection-pytorch

for l in hindi; do
    python "$ROOT"/evalm.py --gold "$ROOT"/data/"$l"-uncovered-test --guess "$ROOT"/preds/"$l"-phone-preds-medium --lang "$l" --phone --print_disagreement
done;
