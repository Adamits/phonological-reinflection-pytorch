import torch
from torch.autograd import Variable

import codecs
import numpy as np

import epitran
import panphon
from util import lang2ISO, get_phones

EOS="<EOS>"
EOS_index = 0
UNK_symbol = "@"
UNK_index = 1

class DistinctiveFeatureData:
    """
    For getting sequences of characters in panphon
    feature representations. And storing the characters
    for lookup during decoding.

    Right now, 3 use cases:

    1. binary vectors, with 1 -> 1 and -1 | 0 -> 0
       to be passed to an embedding mapping
       and returning a single 'embedded' vector

    2. 3 vector settings, 1, 0, and -1 preserved.
       When combined with embedding mapping,
       feature_x as 1 and feature_x as -1 will have
       the same absolute value

    3. binary vectors with feature_vocab * 2 features.
       This is to account for
       feature_x_1 and feature_x_-1 as 2 separate features
       that can be turned on.

    *** To justify cases 2 and 3, I need to look into
        which phonemes may or may not
        benefit from knowledge of the 0, not-applicable setting.***

    """
    def __init__(self, fn, lang):
        """
        We need an output dicitonary mapping each phoneme
        to a unique ID for the decoding portion

        We need an input vector of binary 1/0 (maybe -1 too?)
        representing any feature (including EOS, ' ', PAD as a
        unique feature). This is used for case 2

        We need a method that maps the input vector to a vector
        of unique ids for the features that are 'on'.
        This is for use case 2, wherein every feature is
        its own row in the embedding matrix, to be combined.
        """
        self.fn = fn
        self.lang = lang
        self.epi = epitran.Epitran(lang2ISO(lang))
        self.ft = panphon.FeatureTable()
        self.feature_vocab = self.ft.names
        self.examples, self.tags, self.char_vocab =\
                                        self._prepare_data()
        self.symbol_vocab = [EOS, UNK_symbol] + self.tags
        # Compute indices at which each tag should have 1
        # in the tag one-hot feature vector
        # Note that we do not subtract 1 from feature_vocab
        # to account for ' ' that was added when making the triples
        self.symbols2i = {s: i for i, s in\
                          enumerate(self.symbol_vocab)}
        # char2i is our 'output' dict
        self.char2i = {EOS: EOS_index,\
                       UNK_symbol: UNK_index}
        # i+2 to account for EOS and UNK at 1 and 0
        self.char2i.update({c: i+2 for i, c in
                        enumerate(self.char_vocab)})


    def _read_data(self, fn):
        """
        For the features, we need to make sure not to do
        feature lookups on tags (which don't have any phonology), and
        characters. We therefore override to return a triple
        """
        with codecs.open(fn, "r", encoding='utf-8') as\
             file:
            lines = [l.strip().split('\t') for l in\
                     file]
            triples = [(get_phones(self.epi, lemma),\
                    tags.split(";"), get_phones(self.epi, wf))\
                    for lemma, wf, tags in lines]

            return triples

    def _prepare_data(self):
        pairs = self._read_data(self.fn)
        tag_vocab = []
        feature_vocab = []
        char_vocab = []
        examples = []
        for lemma, tags, wf in pairs:
            for tag in tags:
                if tag not in tag_vocab:
                    tag_vocab.append(tag)
            for c in list(lemma) + list(wf):
                if c not in char_vocab:
                    char_vocab.append(c)
            # A triple of [feature vectors, ...], tag list,
            # word form char list
            examples.append((lemma + ";" + ";".join(tags)\
                             , self.extract_features_array(\
                                    lemma), tags, wf))

        return (examples, tag_vocab, char_vocab)


    def extract_features_array(self, phones):
        # Get the index of all spaces - need to
        # TODO: change to get indices of spaecs between SEGMENTS
        phone_words = phones.split(' ')
        segments = [self.ft.word_array(self.feature_vocab,\
                                       pw) for pw in phone_words]
        # This will find the index for every space in the phones input,
        # In terms of the # of segments (ft model ignores spaces).
        # Each index is actually i less than its true index
        # In order to account for the nature of np.insert
        # With a vector of indices.
        # (e.g inserting at [1, 3, 5] will insert at 1+0, 3+1, 5+2, etc.)
        space_indices = [sum([len(s) for s in segments[:i+1]])\
                         for i, seg in enumerate(segments)][:-1]

        f = self.ft.word_array(self.feature_vocab,\
                                      phones)

        # Change all -1 features in the np array to 0
        f[f < 0] = 0
        # Add a 0 to the end of each feature vector in the
        # np array, in order to account for a blank space (' ')
        # feature
        f = np.insert(f, len(f[0]), [0], axis=1)

        # Add a vector of 0's for space at each space index
        f = np.insert(f, space_indices, [0], axis=0)

        # Update the last feature of each space vector
        # to be 1, to show the 'space-feature' is active.
        # Need this list comprehension jere to get the index
        # Of the now adusted f matrix correct.
        f[[s + i for i, s in enumerate(space_indices)], -1] = 1

        return f

    @classmethod
    def encode_examples(cls, examples, symbols2i, char2i):
        """
        After building the examples, we need to add to each
        feature vector to include tags and extra symbols.

        This entails combining EOS + feature matrix + tags + EOS
        into one input matrix (np array) with a binary feature slot
        appended to each vector for EOS, and every possible tag
        """
        pairs = []

        for input_text, feats, tags, out in examples:
            # Need to make vectors in feats of the
            # full length, given new features
            f = np.append(feats,\
                np.zeros((len(feats), len(symbols2i))), axis=1)

            # tags x tag 'features' (including EOS 'feature')
            tags_matrix = np.zeros((len(tags), len(symbols2i)))
            for i, t in enumerate(tags):
                tags_matrix[i, symbols2i[t]] = 1

            # Add a matrix of 0's to the front of each tag
            # vector in order to account for all other features
            tags_matrix = np.append(np.zeros((tags_matrix.shape[0],\
                                feats.shape[1])), tags_matrix, axis=1)

            EOS_vector = np.zeros((1, f.shape[1]))
            EOS_vector[0, symbols2i[EOS]] = 1

            # Put them all together to make the full sequence matrix
            inp = np.append(f, tags_matrix, axis=0)
            inp = np.append(inp, EOS_vector, axis=0)
            inp = np.append(EOS_vector, inp, axis=0)

            i_tensor = Variable(torch.FloatTensor(inp))
            o_tensor = Variable(torch.LongTensor(\
            [char2i[EOS]] + [char2i.get(o, UNK_index) for o in out] +\
                                        [char2i[EOS]]))
            input_text = EOS + input_text + EOS
            out = EOS + out + EOS

            # Add the tuple to the pairs list
            pairs.append((input_text, i_tensor,\
                          out, o_tensor))

        return pairs
