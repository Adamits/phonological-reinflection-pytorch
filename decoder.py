import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, hidden_size_input, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size_input = hidden_size_input
        # MLP to run over encoder_outputs
        self.attn1 = nn.Linear(self.hidden_size_input +\
                               self.hidden_size, self.hidden_size)
        self.attn2 = nn.Linear(self.hidden_size, 1)

    def forward(self, hidden, encoder_outputs, mask):
        """
        Compute the attention distribution from the encoder outputs
        at all timesteps, and the previous hiddens tate in the GRU.
        """
        # Make them both B x seq_length x H
        H = hidden.repeat(encoder_outputs.size(0), 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        
        # Get the scores of each time step in the output
        attn_scores = self.score(H, encoder_outputs)
        # Mask the scores with -inf at each padded character
        # So that softmax computes a 0 towards the distribution
        # For that cell.
        attn_scores.data.masked_fill_(mask, -float('inf'))
        # Return the attention distribution
        # B x 1 x seq_len
        return F.softmax(attn_scores).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        """
        Compute the scores by running the encoder outputs
        through the MLP, and taking dot product with previous
        hidden state of the GRU
        """
        tanh = torch.nn.Tanh()
        # Concat and run through Linear layer,
        # applying a non-linearity
        # tanh(X * W + b), giving B x H x seq_len
        attn1_output = tanh(self.attn1(torch.cat([encoder_outputs,\
                                      hidden], 2)))
        # Run through second Linear layer
        # W * attn1_output + b
        scores = self.attn2(attn1_output)
        # return B x seq_len
        return scores.squeeze(2)
        

class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size,\
            output_size, bidirectional_input=False):
        super(Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size_input = hidden_size * 2 if\
                bidirectional_input else\
               hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional_input = bidirectional_input

        # Hidden state whose parameters are shared across all examples
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))
        self.embedding = nn.Embedding(self.output_size\
                , self.embedding_size)
        self.gru = nn.GRU(self.hidden_size_input +\
                self.embedding_size, self.hidden_size)
        self.attn = Attn(self.hidden_size_input, self.hidden_size)
        # MLP for mapping GRU output to a
        # distribution over characters
        self.out = nn.Linear(self.hidden_size,\
                             self.output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, last_hidden, encoder_out,\
                batch_size, use_cuda, mask):
        # Reshape to 1 x B x H (double check this
        # is the right way to do this...)
        embedded = self.embedding(input).view(1,\
                    batch_size, self.hidden_size)

        #if self.bidirectional_input:
        #    last_encoder_out = encoder_out[0, :, :].unsqueeze(0)
        #else:
        #    last_encoder_out = encoder_out[-1, :, :].unsqueeze(0)

        # Get the attn distribution over enc outputs of
        # B x 1 x seq_len
        attn_weights = self.attn(last_hidden, encoder_out, mask)
        # Apply attention to the enc outputs
        context = torch.bmm(attn_weights, encoder_out.transpose(0, 1))
        # B x 1 x H --> 1 x B x H, In order ot match embedded dims
        context = context.transpose(0, 1)
        # Concatenate the embedding and
        # last encoder output to pass to the GRU
        gru_in = torch.cat((embedded, \
                            context), 2)
        # Run the GRU, also passing it the
        # h of the previous GRU run
        output, hidden = self.gru(gru_in, last_hidden)

        # Run it through the MLP
        output = self.out(output)
        # Compute log_softmax scores for NLLLoss
        scores = self.softmax(output)

        return scores, hidden

    def init_hidden(self, batch_size):
        # This will copy the h0 tensor B times to get a 1 x B x H tensor
        return self.h0.repeat(1, batch_size, 1)
