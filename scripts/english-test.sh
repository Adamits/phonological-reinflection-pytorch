#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

python "$ROOT"/test.py "$ROOT"/data/english-uncovered-test "$ROOT"/preds "$ROOT"/models/encoder-english-text "$ROOT"/models/decoder-english-text "$ROOT"/models/char2i-english-text.pkl english text 100 --gpu
