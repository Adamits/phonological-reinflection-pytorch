import random
import argparse

from encoder_decoder import *
from data_prep import *

use_cuda = torch.cuda.is_available()

EOS="<EOS>"
EOS_index=0

def indexesFromSentence(sentence, char2i):
    return [char2i[c] for c in sentence]

def variableFromSentence(sentence, char2i):
    indexes = indexesFromSentence(sentence, char2i)
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

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, max_length):
    """
    Compute the loss and make the parameter updates for a single sequence,
    where loss is the average of losses for each in the sequence
    """
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[EOS_index]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        # I believe that this is just taking the index with the highest
        # value in the softmax distribution
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        loss += loss_function(decoder_output, target_variable[di])
        if ni == EOS_index:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def trainIters(encoder, decoder, pairs, n_iters, print_every=25, learning_rate=0.01, max_length):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    loss_function = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, loss_function, max_length)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train the encoder decoder with reinflection data')
    parser.add_argument('filename', metavar='fn',
                        help='the filename of the file to train on')
    parser.add_argument('lang', metavar='lang',
                        help='The language that we are training on')

    args = parser.parse_args()
    hidden_size = 500
    data = DataPrep(args['filename'])
    encoder1 = EncoderRNN(len(data.input_vocab), hidden_size)
    attn_decoder1 = AttnDecoderRNN(hidden_size, len(data.output_vocab), dropout_p=0.1)

    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    trainIters(encoder1, attn_decoder1, data.pairs, 75000, print_every=5000)

    torch.save(encoder1.state_dict(), "./models/%s-encoder" % args['lang'])
    torch.save(attn_decoder1.state_dict(), "./models/%s-decoder" % args['lang'])
