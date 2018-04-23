#! /bin/bash
ROOT=~/phonological-reinflection-pytorch
l=german

python "$ROOT"/test.py "$ROOT"/data/"$l"-uncovered-test "$ROOT"/preds "$ROOT"/data/german-train-medium "$ROOT"/data/german-dev german medium 30 .08 1.5
