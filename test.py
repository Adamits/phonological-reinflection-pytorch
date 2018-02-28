import argparse

from encoder_decoder import *
from data_prep import *

def predict(encoder, decoder, sentence, max_length):
    input_variable = variableFromSentence(sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[EOS_index]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_index:
                decoded_words.append(EOS)
                break
            else:
                decoded_words.append(outp.index2word[ni])

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input


    return decoded_words, decoder_attentions[:di + 1]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Test the encoder decoder with reinflection data')
    parser.add_argument('filename', metavar='fn',
                        help='the filename of the file to train on')
    parser.add_argument('encoderModel', metavar='model',
                        help='The encoder that we are evaluating on')
    parser.add_argument('decoderModel', metavar='model',
                        help='The decoder that we are evaluating on')

    args = parser.parse_args()
    hidden_size = 500
    test_data = DataPrep(args['filename'])
    encoder = torch.load('./models/' % args['encoderModel'])
    decoder = torch.load('./models/' % args['decoderModel'])

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    for isentence, osentence in test_data.pairs:
        pred, attentions = predicts(encoder, decoder, isentence, MAX_LENGTH)
        print("===============")
        print("ATTENTION")
        print(attentions)
        print("PREDICTION")
        print(pred)
        print("GOLD")
        print(osentence)
