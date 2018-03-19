import codecs

import torch
from torch.autograd import Variable

#import epitran
#import panphon
#from util import lang2ISO

# Remember to add this to char2i
EOS="<EOS>"
EOS_index=0

class DataPrep:
    def __init__(self, fn):
        self.fn = fn
        self.char2i = {EOS: EOS_index}
        self.pairs, self.input_vocab, self.output_vocab = self.prepareData(self.fn)
        # Assign indices for all chars (or tags) from both vocabs,
        # starting at 1 to account for EOS
        self.char2i.update({c: i+1 for i, c in enumerate(list(set(self.input_vocab + self.output_vocab)))})

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
        #self.epi = epitran.Epitran(lang2ISO(lang))

    def readData(self, fn):
        print("Reading lines...")

        # Read the file and split into lines
        with codecs.open(fn, "r", encoding='utf-8') as file:
            lines = [l.strip().split('\t') for l in file]

        # Split every line into pairs
        # Form of [[e, a, c, h, " ", c, h, a, r, tag, tag, tag], w, o, r, d, " ", f, o, r, m]
        #pairs = [(list(lemma) + tags.split(";"), list(wf)) for lemma, wf, tags in lines]
        #lemmas = [get_phones(epi, lemma) for lemma, _, _ in data]
        #wfs = [get_phones(epi, wf) for _, wf, _ in data]

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


class Batch():
    """
    A single batch of data
    """
    def __init__(self, data):
        self.symbol = EOS
        self.size = (len(data))
        self.input = [d[0] for d in data]
        self.output = [d[1] for d in data]
        self.input_lengths = [len(i) for i in self.input]
        # Because it is sorted, first input in the batch should have max_length
        self.max_length_in = max(self.input_lengths)
        self.output_lengths = [len(o) for o in self.output]
        # We do not expect the outputs to be sorted though... (but we can expect that they might
        # have SOME similarity in length to input)
        self.max_length_out = max(self.output_lengths)

    def input_variable(self, char2i, use_cuda):
        """
        Turn the input into a tensor of batch_size x batch_length

        Returns the input
        """
        tensor = torch.LongTensor(self.size, self.max_length_in)

        for i, word in enumerate(self.input):
            # Arbitrarily return character with index 5
            # If we have not seen this input
            ids = [char2i.get(c, 5) for c in word]
            # Pad the difference with symbol
            ids = ids + [char2i[self.symbol]] * (self.max_length_in - len(word))
            tensor[i] = torch.LongTensor(ids)

        self.input = Variable(tensor).cuda() if use_cuda else Variable(tensor)
        return self.input

    def output_variable(self, char2i, use_cuda):
        """
        Turn the output into a tensor of batch_size x batch_length

        Returns the output
        """
        tensor = torch.LongTensor(self.size, self.max_length_out)

        for i, word in enumerate(self.output):
            ids = [char2i.get(c, 5) for c in word]
            # Pad the difference with symbol
            ids = ids + [char2i[self.symbol]] * (self.max_length_out - len(word))
            tensor[i] = torch.LongTensor(ids)

        self.output = Variable(tensor).cuda() if use_cuda else Variable(tensor)
        return self.output

def get_batches(pairs, batch_size, char2i, use_cuda):
    """
    Returns a list of batch objects for the entire set of paris.
    """
    sorted_pairs = pairs.copy()

    # Sort by input length so that samples in the same batch have similar len
    sorted_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    # Split sorted_data into n batches each of size batch_length
    batches = [sorted_pairs[i:i+batch_size] for i in range(0, len(sorted_pairs), batch_size)]
    # Loop over indices so we can modify batches in place
    for i in range(len(batches)):
        batches[i] = Batch(batches[i])
        batches[i].input_variable(char2i, use_cuda)
        batches[i].output_variable(char2i, use_cuda)

    return batches

def add_EOS_to_pair(pairs):
    eos_pairs = []

    for inp, outp in pairs:
        eos_pairs.append(([EOS] + inp + [EOS],
                          [EOS] + outp + [EOS]))
    return eos_pairs

def indexesFromSentence(sentence, char2i):
    """
    return the list of indices, skipping unknown chars
    """
    return [char2i.get(c, 5) for c in sentence]

def variableFromSentence(sentence, char2i, use_cuda):
    indexes = indexesFromSentence(sentence, char2i)
    indexes = [EOS_index] + indexes + [EOS_index]
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair, char2i, use_cuda):
    input_variable = variableFromSentence(pair[0], char2i, use_cuda)
    target_variable = variableFromSentence(pair[1], char2i, use_cuda)
    return (input_variable, target_variable)
