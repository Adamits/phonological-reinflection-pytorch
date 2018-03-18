import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

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
        #packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(embedded, hidden)
        #outputs, output_length = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        return outputs, hidden

    def initHidden(self, batch_size, use_cuda):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.flatten = nn.Parameter(torch.FloatTensor(1, self.hidden_size))
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input, hidden, encoder_outputs, use_cuda):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Get the embedding of the current input word (last output word)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)

        # Create variable to store attention energies for the batch
        attn_energies = Variable(torch.zeros(batch_size, max_len))
        if use_cuda: attn_energies = attn_energies.cuda()
        # Attend over the encoder outputs and last hidden state
        attn_energies = self._attend(encoder_outputs, hidden, batch_size, max_len, attn_energies)

        # Normalize to get the actual weights,
        # Resize softmax to 1 x 1 x seq_len
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)

        # 'Apply' attention by taking weights * encoder outputs
        context = torch.bmm(attn_weights,
                            encoder_outputs.transpose(0, 1))

        context = context.transpose(0, 1)
        output = torch.cat((embedded, context), 2)

        output, hidden = self.gru(output, hidden)

        output = output.squeeze(0)
        context = context.squeeze(0)

        # Decoder softmax over the result of running
        # the concatenation of the decoder RNN output
        # and the attn_wights applied to encoder outputs
        # Throught the output MLP
        output = F.log_softmax(self.out(torch.cat((output, context), 1)), dim=1)

        return output, hidden, attn_weights

    def _attend(self, encoder_outputs, hidden, batch_size, max_len, attn_energies):
        """
        Takes the output of the encoder over the entire sequence, the last hidden state,
        the sizes, and an empty energies matrix. Computes the score, or attention 'energy'
        for the entire batch, for the entire sequence
        """
        def _score(h, encoder_output):
            # Run the attn MLP over the concat of a hidden state and
            # encoder_output, along axis 1
            energy = self.attn(torch.cat((h, encoder_output), 1))
            # Simple dot product of hidden and energy
            energy = h.view(-1).dot(energy.view(-1))
            return energy

        # Calculate energies for each encoder output by applying linear attn layer
        # to concat of the (last) hidden state and each encoder output
        for b in range(batch_size):
            for i in range(max_len):
                # Need to unsqueeze each output to have rank 3 tensor,
                # and apply concat along axis 1.
                # DO WE NEED SOME MASKING MECHANISM SO THAT PADDING DOES NOT GET AN ENERGY?
                attn_energies[b, i] = _score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        return attn_energies
