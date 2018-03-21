import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import math

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        # initialize all nn.Module attributes
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # Set embedding matrix to find weights from input to hidden layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        # 'gated recurrent unit' RNN layer
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input_batch, input_lengths, hidden=None):
        embedded = self.embedding(input_batch)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_length = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional outputs
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

    def initHidden(self, batch_size, use_cuda):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        # hidden * 2 for concatenated bidirecitonal outputs from encoder
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        # Try a different imp
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
                        

    def forward(self, hidden, encoder_outputs, mask):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        # H = hidden.unsqueeze(1)
        encoder_outputs = encoder_outputs.transpose(0,1) # transpose for [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        attn_energies.data.masked_fill_(mask, -float('inf'))
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        # Apply attention MLP over hidden mat ^ encoder_outputs mat
        # energy = F.tanh(self.attn(encoder_outputs)) # [B*T*H]
        # print(hidden.size(), encoder_outputs.size())
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*2*3H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        # print(energy.size())
        # Trying other imp
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        # print(energy.size())
        return energy.squeeze(1) #[B*T]
        # Should be synonymous to dot product of each corresponding output and hidden.
        # energy = torch.bmm(hidden, energy) # [B*1*T]
        # return energy.squeeze(1) #[B*T]


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        # Share h0 across all training examples
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = Attention(hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # * 3 for bidirectional encoder outputs concatenated
        # which will be concatenated to previous decoder embedding
        # Each of which is of hidden size
        self.gru = nn.GRU(self.hidden_size * 3, self.hidden_size, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, last_hidden, encoder_outputs, use_cuda, mask):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(input).view(1, batch_size, self.hidden_size) # [1 x B x H]
        embedded = self.dropout(embedded)

        # Get the attn weights from the Attention modeoutputs
#        attn_weights = self.attn(hidden[-1], encoder_outputs, mask)
#        print("MASKING AND ATTN")
#        print(mask.size())
#        print(attn_weights[0])
#        print(attn_weights[0].size())
        # 'Apply' attention by taking weights * encoder outputs
        # B x 1 x H
#        context = torch.bmm(attn_weights,
#                            encoder_outputs.transpose(0, 1))
#        context = context.transpose(0, 1) # Make it [1 x B x H]

        # Concat the embedded input char and the 'context' to be run through RNN
#        output = torch.cat((embedded, context), 2)

        last_encoder_output= encoder_outputs[-1].unsqueeze(0)
        output = torch.cat((embedded, last_encoder_output), 2)
        output, hidden = self.gru(output, last_hidden)
        # Make them both just B x H
        output = output.squeeze(0)
#        context = context.squeeze(0)

        # Decoder softmax over the result of running
        # the concatenation of the decoder RNN output
        # and the attn_wights applied to encoder outputs
        # Through the output MLP
#        output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim=1)
        output = F.log_softmax(self.out(output), dim=1)
        return output, hidden

    def initHidden(self, batch_size):
        # Copy the h0 tensor B times to get a 1 x B x H vector
        return self.h0.repeat(1, batch_size, 1)
