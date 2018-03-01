import codecs

import torch
from torch.autograd import Variable

import epitran
import panphon
from util import lang2ISO

# Remember to add this to char2i
EOS="<EOS>"
EOS_index=0

use_cuda = False

class DataPrep:
    def __init__(self, fn):
        self.fn = fn
        self.char2i = {EOS: EOS_index}
        self.pairs, self.input_vocab, self.output_vocab = self.prepareData(self.fn)
        # Assign indices for all chars (or tags) from both vocabs,
        # starting at 1 to account for 0: EOS
        self.char2i = {c: i+1 for i, c in enumerate(list(set(self.input_vocab + self.output_vocab)))}

    def readData(self, fn):
        print("Reading lines...")

        # Read the file and split into lines
        with codecs.open(fn, "r", encoding='utf-8') as file:
            lines = [l.strip().split('\t') for l in file]

        # Split every line into pairs
        # Form of [[e, a, c, h, " ", c, h, a, r, tag, tag, tag], w, o, r, d, " ", f, o, r, m]
        pairs = [(list(lemma) + tags.split(";"), list(wf)) for lemma, wf, tags in lines]

        return pairs

    def prepareData(self, fn):
        pairs = self.readData(fn)
        print("Counting chars...")
        input_vocab = []
        output_vocab = []
        for pair in pairs:
            inp, outp = pair
            for c in inp:
                if c not in input_vocab:
                    input_vocab.append(c)
            for c in outp:
                if c not in output_vocab:
                    output_vocab.append(c)

        return pairs, input_vocab, output_vocab

class DataPrepPhones(DataPrep):
    def __init__(self, fn, lang):
        super(DataPrepPhones, self).__init__()
        self.epi = epitran.Epitran(lang2ISO(lang))

    def readData(self, fn):
        print("Reading lines...")

        # Read the file and split into lines
        with codecs.open(fn, "r", encoding='utf-8') as file:
            lines = [l.strip().split('\t') for l in file]

        # Split every line into pairs
        # Form of [[e, a, c, h, " ", c, h, a, r, tag, tag, tag], w, o, r, d, " ", f, o, r, m]
        pairs = [(list(lemma) + tags.split(";"), list(wf)) for lemma, wf, tags in lines]
        lemmas = [get_phones(epi, lemma) for lemma, _, _ in data]
        wfs = [get_phones(epi, wf) for _, wf, _ in data]

        return pairs

    def prepareData(self, fn):
        pairs = self.readData(fn)
        print("Counting chars...")
        input_vocab = []
        output_vocab = []
        for pair in pairs:
            inp, outp = pair
            for c in inp:
                if c not in input_vocab:
                    input_vocab.append(c)
            for c in outp:
                if c not in output_vocab:
                    output_vocab.append(c)

        return pairs, input_vocab, output_vocab

class DataPrepPhoneFeatures(DataPrep):
    def __init__(self):
        super(DataPrepPhoneFeatures, self).__init__()


def indexesFromSentence(sentence, char2i):
    return [char2i[c] for c in sentence]

def variableFromSentence(sentence, char2i):
    indexes = indexesFromSentence(sentence, char2i)
    indexes.insert(0, EOS_index)
    indexes.append(EOS_index)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair, char2i):
    input_variable = variableFromSentence(pair[0], char2i)
    target_variable = variableFromSentence(pair[1], char2i)
    return (input_variable, target_variable)
