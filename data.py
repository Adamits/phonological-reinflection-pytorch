import torch
from torch.autograd import Variable

import codecs
import numpy as np

import epitran
import panphon
from util import lang2ISO, get_phones, get_phone_segments

EOS="<EOS>"
EOS_index = 0
PAD_symbol = "#"
PAD_index = 1
UNK_symbol = "@"
UNK_index = 2

class Data:
    def __init__(self, fn):
        self.fn = fn
        self.char2i = {EOS: EOS_index,\
                       PAD_symbol: PAD_index,\
                       UNK_symbol: UNK_index}
        self.pairs, self.vocab = self._prepare_data()

    def _read_data(self, fn):
        with codecs.open(fn, "r", encoding='utf-8') as\
             file:
            lines = [l.strip().split('\t') for l in\
                     file]
            pairs = [(list(lemma) + tags.split(";"),\
                list(wf)) for lemma, wf, tags in lines]

            return pairs

    def add_to_dict(self, vocab, size):
        self.char2i.update({c: i+size for i, c in\
                            enumerate(list(vocab))})

    def _prepare_data(self):
        pairs = self._read_data(self.fn)
        vocab = []
        for pair in pairs:
            inp, outp = pair
            for c in inp + outp:
                if c not in vocab:
                    vocab.append(c)

        self.add_to_dict(vocab, len(self.char2i))
        return pairs, vocab


class PhoneData(Data):
    def __init__(self, fn, lang, segment_phone=False):
        self.fn = fn
        self.epi = epitran.Epitran(lang2ISO(lang))
        self.segment_phone = segment_phone
        super(PhoneData, self).__init__(self.fn)

    def _get_phones(self, epi, text):
        if self.segment_phone:
            return get_phone_segments(epi, text)
        else:
            return list(get_phones(epi, text))
        
    def _read_data(self, fn):
        with codecs.open(fn, "r", encoding='utf-8') as\
             file:
            lines = [l.strip().split('\t') for l in\
                 file]
            pairs = [(self._get_phones(self.epi, lemma.lower()) +\
                      tags.split(";"), self._get_phones(self.epi, wf.lower()))\
                     for lemma, wf, tags in lines]

        return pairs

    def tensor_pairs(self, char2i):
        tensor_pairs = []
        for inp, out in self.pairs:
            eos_inp = [char2i.get(c, UNK_index) for\
                         c in [EOS] + inp + [EOS]]
            eos_out = [char2i.get(c, UNK_index) for\
                       c in [EOS] + out + [EOS]]
            i_tensor = Variable(torch.LongTensor(eos_inp))
            o_tensor = Variable(torch.LongTensor(eos_out))
            tensor_pairs.append((i_tensor, o_tensor))

        return tensor_pairs

class Batch(object):
    def __init__(self, pairs, symbol):
        self.pairs = pairs
        self.inputs = [p[0] for p in pairs]
        self.outputs = [p[1] for p in pairs]
        self.lengths_in = [len(i) for i in self.inputs]
        self.lengths_out = [len(i) for i in\
                            self.outputs]
        self.max_length_in = max(self.lengths_in)
        self.max_length_out = max(self.lengths_out)
        self.symbol = symbol
        self.size = len(self.inputs)
        self.input_variable = None
        self.output_variable = None
        self.input_mask = None

    def make_input_variable(self, char2i, use_cuda):
        """
        Turn the input into a tensor of
        batch_size x batch_length

        Returns the input
        """

        tensor = torch.LongTensor(self.size,\
                                  self.max_length_in)

        for i, word in enumerate(self.inputs):
            ids = [char2i.get(c, UNK_index) \
                   for c in word]
            # Pad the difference with symbol
            ids = ids + [char2i[self.symbol]] *\
                  (self.max_length_in - len(word))
            tensor[i] = torch.LongTensor(ids)

        input_var = Variable(tensor).cuda() if\
                    use_cuda else Variable(tensor)
        self.input_variable = input_var
        return input_var

    def make_output_variable(self, char2i, use_cuda):
        """
        Turn the output into a tensor
        of batch_size x batch_length

        Returns the output variable
        """
        tensor = torch.LongTensor(self.size,\
                                  self.max_length_out)

        for i, o in enumerate(self.outputs):
            ids = [char2i.get(c, UNK_index) \
                   for c in o]

            for c in o:
                if char2i.get(c, UNK_index) == UNK_index:
                    print("Unknown from output variable: %s" % c)
                
            # Pad the difference with symbol
            ids = ids + [char2i[self.symbol]] *\
                  (self.max_length_out - len(o))
            tensor[i] = torch.LongTensor(ids)

        output_var = Variable(tensor).cuda() if\
                     use_cuda else Variable(tensor)
        self.output_variable = output_var
        return output_var

    def make_masks(self, use_cuda):
        """
        Only the input needs to be masked
        """
        mask_type = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
        seq_range = torch.range(0, self.max_length_in - 1).long()
        seq_range_expanded = seq_range.unsqueeze(0).expand(self.size, self.max_length_in)

        seq_range_expanded = seq_range_expanded
        seq_range_expanded = seq_range_expanded.cuda() if use_cuda else seq_range_expanded

        input_lengths = torch.LongTensor(self.lengths_in)
        input_lengths = input_lengths.cuda() if use_cuda else input_lengths

        seq_length_expanded = (input_lengths.unsqueeze(1)\
                               .expand_as(seq_range_expanded))

        mask_mat = seq_range_expanded < seq_length_expanded
        mask_mat = 1-mask_mat.long()

        mask_mat = mask_mat.type(mask_type)
        self.input_mask = mask_mat
        return mask_mat

class FeatureBatch(object):
    def __init__(self, triples, symbol):
        self.triples = triples
        self.input_chars = [t[0] for t in triples]
        self.input_tags = [t[1] for t in triples]
        self.outputs = [t[2] for t in triples]
        self.lengths_in = [len[i] + len[j] for i, j in\
                           zip(self.input_chars, self.input_tags)]
        self.lengths_out = [len(i) for i in self.outputs]
        self.feature_indices = [i for i, c in\
                enumerate(self.input_chars) if c not in [' ', "<EOS>"]]
        self.max_length_in = max(self.lengths_in)
        self.max_length_out = max(self.lengths_out)
        self.symbol = symbol
        self.size = len(self.triples)
        self.input_variable = None
        self.output_variable = None
        self.input_mask = None

        def make_input_variable(self, char2i, use_cuda):
            """
            Turn the input into a tensor of
            batch_size x batch_length of indices, usng a dummy
            UNK index in place of characters that will be replaced
            by features

            Returns the input
            """

            tensor = torch.LongTensor(self.size,\
                                      self.max_length_in)

            for i, word in enumerate(self.input_chars + self.input_tags):
                ids = [char2i.get(c[0], UNK_index) \
                    for c in word if len(c) == 1]
                # Pad the difference with symbol
                ids = ids + [char2i[self.symbol]] *\
                      (self.max_length_in - len(word))
                tensor[i] = torch.LongTensor(ids)

            input_var = Variable(tensor).cuda() if\
                        use_cuda else Variable(tensor)
            self.input_variable = input_var
            return input_var

    def make_output_variable(self, char2i, use_cuda):
            """
            Turn the output into a tensor
            of batch_size x batch_length

            Returns the output variable
            """
            tensor = torch.LongTensor(self.size,\
                                      self.max_length_out)

            for i, o in enumerate(self.outputs):
                ids = [char2i.get(c, UNK_index) \
                       for c in o]
                # Pad the difference with symbol
                ids = ids + [char2i[self.symbol]] *\
                      (self.max_length_out - len(o))
                tensor[i] = torch.LongTensor(ids)

            output_var = Variable(tensor).cuda() if\
                         use_cuda else Variable(tensor)

            self.output_variable = output_var
            return output_var

    def make_masks(self, use_cuda):
            """
            Only the input needs to be masked
            """
            mask_type = torch.cuda.ByteTensor\
                        if use_cuda else torch.ByteTensor
            seq_range = torch.range(0, self.max_length_in - 1).long()
            seq_range_expanded = seq_range.unsqueeze(0\
            ).expand(self.size, self.max_length_in)

            seq_range_expanded = seq_range_expanded
            seq_range_expanded = seq_range_expanded.cuda()\
                                 if use_cuda else seq_range_expanded

            input_lengths = torch.LongTensor(self.lengths_in)
            input_lengths = input_lengths.cuda()\
                            if use_cuda else input_lengths

            seq_length_expanded = (input_lengths.unsqueeze(1)\
                                   .expand_as(seq_range_expanded))

            mask_mat = seq_range_expanded < seq_length_expanded
            mask_mat = 1-mask_mat.long()

            mask_mat = mask_mat.type(mask_type)
            self.input_mask = mask_mat
            return mask_mat

def get_feature_batches(triples, batch_size,\
                char2i, PAD_symbol, use_cuda, test_mode=False):
    sorted_pairs = triples.copy()
    # Need to retain original order if we are in
    # Test mode
    if not test_mode:
        sorted_pairs.sort(key=lambda x: len(x[0] + x[1]),\
                          reverse=True)

    batches = [Batch(sorted_pairs[i:i+batch_size],\
                     PAD_symbol) for i in range(0,\
                    len(sorted_pairs), batch_size)]

    for batch in batches:
        batch.make_input_variable(char2i, use_cuda)
        batch.make_output_variable(char2i, use_cuda)
        batch.make_masks(use_cuda)

    return batches

def get_batches(pairs, batch_size,\
                char2i, PAD_symbol, use_cuda, test_mode=False):
    sorted_pairs = pairs.copy()
    # Need to retain original order if we are in
    # Test mode
    if not test_mode:
        sorted_pairs.sort(key=lambda x: len(x[0]),\
                          reverse=True)

    batches = [Batch(sorted_pairs[i:i+batch_size],\
                     PAD_symbol) for i in range(0,\
                    len(sorted_pairs), batch_size)]

    for batch in batches:
        batch.make_input_variable(char2i, use_cuda)
        batch.make_output_variable(char2i, use_cuda)
        batch.make_masks(use_cuda)

    return batches

def _read_data(fn):
    print("Reading lines...")
    # Read the file and split into lines
    with codecs.open(fn, "r", encoding='utf-8') as file:
        lines = [l.strip().split('\t') for l in file]

    return lines

def _get_pairs(lines, lang):
    return [(list(wf), lang) for lemma, wf, tags in lines]

def _get_vocab(lines):
    chars = [c for lemma, wf, tags in lines for c in list(wf)]
    return list(set(chars))

def get_data(fn, lang):
    """
    Returns a tuple of input/lang_name text pairs, and a vocab list
    """
    lines = _read_data(fn)
    return (_get_pairs(lines, lang), _get_vocab(lines))

def wf2Var(wf, char2i):
    # If c is not in the vocab, return UNK
    ids = [char2i.get(c, UNK_index) for c in wf]
    result = Variable(torch.LongTensor(ids))

    return result

def label2Var(label, label2i):
    l_id = [label2i[label]]
    result = Variable(torch.LongTensor(l_id))

    return result
