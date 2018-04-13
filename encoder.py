import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, bidirectional=False):
        super(Encoder, self).__init__()
        # Size of the input vocab
        self.input_size = input_size
        # dims of each embedding
        self.embedding_size = embedding_size
        # dims of the hidden state
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, bidirectional=bidirectional)

    def forward(self, batch):
        """ 
        Lookup the embedded representation
        of the inputs
        """
        embedded = self.embedding(batch)
        # Run through the GRU
        """
        GRU expects input of:
        (seq_len, batch_len, embedding_size)
        """
        output, hidden = self.gru(embedded)
        return output, hidden
