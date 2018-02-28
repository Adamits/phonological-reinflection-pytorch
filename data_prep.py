import codecs

import torch
from torch.autograd import Variable

# Remember to add this to char2i
EOS="<EOS>"
EOS_index=0

class DataPrep:
    def __init__(fn):
    self.char2i = {EOS: EOS_index}
    self.pairs self.input_vocab, self.output_vocab = prepare_data(fn)
    # Assign indices for all chars (or tags) from both vocabs,
    # starting at 1 to account for 0: EOS
    self.char2i = {c: i+1 for c in self.input_vocab + self.output_vocab}

    def readData(fn):
        print("Reading lines...")

        # Read the file and split into lines
        with codecs.open(fn, "r", encoding='utf-8') as file:
            lines = [l.strip().split('\t') for l in file]

        # Split every line into pairs
        # Form of [[e, a, c, h, " ", c, h, a, r, tag, tag, tag], w, o, r, d, " ", f, o, r, m]
        pairs = [(list(lemma) + tags.split(";"), list(wf)) for lemma, wf, tags in lines]

        return pairs

    def prepareData(fn):
        pairs = readData(fn)
        print("Counting chars...")
        for pair in pairs:
            inp, outp = pair
            input_vocab.append([c for c in inp if inp not in input_vocab])
            output_vocab.append([c for c in inp if outp not in output_vocab])

        return pairs, input_vocab, output_vocab

class DataPrepPhones(DataPrep):
    def __init__():
        super(DataPrepPhones, self).__init__()

class DataPrepPhoneFeatures(DataPrep):
    def __init__():
        super(DataPrepPhoneFeatures, self).__init__()

