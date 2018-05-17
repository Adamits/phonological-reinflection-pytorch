import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

torch.set_num_threads(1)

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

class PhonologyEncoder(nn.Module):
    def __init__(self, input_size, embedding_size,\
                 hidden_size, bidirectional=False,\
                 concat_phone=False, phone_input_size=None):
        super(PhonologyEncoder, self).__init__()
        # Size of the input vocab
        self.input_size = input_size
        self.concat_phone = concat_phone
        # dims of each embedding
        self.embedding_size = embedding_size
        # dims of the hidden state
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedding = nn.Parameter(torch.FloatTensor(self.input_size,\
                                        self.embedding_size).normal_())
        if concat_phone:
            self.phone_embedding = nn.Embedding(phone_input_size,\
                                        self.embedding_size)
            self.gru = nn.GRU(self.embedding_size*2, self.hidden_size,\
                          bidirectional=bidirectional)
        else:
            self.gru = nn.GRU(self.embedding_size, self.hidden_size,\
                              bidirectional=bidirectional)

    def forward(self, inp, phone_inp=None):
        """
        inpt: seq_len x num_features matrix
        """
        # matrix matrix product, gives us seq_len x embedding_size
        embedded = inp.mm(self.embedding).view(inp.size()[0], 1, -1)
        if phone_inp is not None:
            if inp.size()[0] != phone_inp.size()[0]:
                raise Exception('you have a problem, phone input has size: %i, and the feature input has size: %i' %\
                                (phone_inp.size()[0], inp.size()[0]))

            phone_embedded = self.phone_embedding(phone_inp).view(\
                                                len(phone_inp), 1, -1)
            embedded = torch.cat((embedded, phone_embedded), 2)
        # Run it through GRU
        output, hidden = self.gru(embedded)
        
        return output, hidden

                
