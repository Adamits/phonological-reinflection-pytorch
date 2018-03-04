import random
import argparse
import pickle

from encoder_decoder import *
from data_prep import *

EOS="<EOS>"
EOS_index=0


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, use_cuda,  max_length=50):
    """
    Compute the loss and make the parameter updates for a single sequence,
    where loss is the average of losses for each in the sequence
    """
    encoder_hidden = encoder.initHidden(use_cuda)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[EOS_index]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    print("INPUT SEQUENCE")
    print(input_variable)
    # SKIP THE FIRST ONE IN THE TARGET AS IT IS EOS AND WE INITIALIZE WITH EOS
    for di in range(1, target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)

        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        print("PRED")
        print(ni)
        print("GOLD")
        print(target_variable[di].data[0])
        print("=========================")

        loss += loss_function(decoder_output, target_variable[di])
        if ni == EOS_index:
            print("BREAKING!!! %i" % ni)
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def trainIters(encoder, decoder, pairs, char2i, n_iters, use_cuda, print_every=100, learning_rate=0.01, max_length=50):
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs), char2i, use_cuda)
                      for i in range(n_iters)]
    loss_function = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]

        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, loss_function, use_cuda, max_length)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('LOSS: %.4f' % print_loss_avg)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train the encoder decoder with reinflection data')
    parser.add_argument('filename', metavar='fn',
                        help='the filename of the file to train on')
    parser.add_argument('lang', metavar='lang',
                        help='The language that we are training on')
    parser.add_argument('--gpu',action='store_true',  help='tell the system to use a gpu if you have cuda set up')

    args = parser.parse_args()
    hidden_size = 500
    data = DataPrep(args.filename)
    input_size = len(data.char2i.keys())
    encoder1 = EncoderRNN(input_size+1, hidden_size)
    attn_decoder1 = AttnDecoderRNN(hidden_size, input_size+1, dropout_p=0.4)

    char2i = data.char2i
    # Store the character dictionary for use in testing
    char_output = open('./models/%s-char2i.pkl' % args.lang, 'wb')
    pickle.dump(char2i, char_output)

    use_cuda = args.gpu

    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    trainIters(encoder1, attn_decoder1, data.pairs, char2i, 75000, use_cuda,  print_every=100)

    torch.save(encoder1, "./models/%s-encoder" % args.lang)
    torch.save(attn_decoder1, "./models/%s-decoder" % args.lang)
