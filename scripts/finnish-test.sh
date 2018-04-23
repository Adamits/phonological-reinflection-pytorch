#! /bin/bash
ROOT=~/phonological-reinflection-pytorch

python "$ROOT"/test.py "$ROOT"/data/finnish-uncovered-test "$ROOT"/preds "$ROOT"/models/encoder-finnish "$ROOT"/models/acceptor-finnish "$ROOT"/models/char2i-finnish.pkl finnish text 100 --gpu
