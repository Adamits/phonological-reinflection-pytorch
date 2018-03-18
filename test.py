import argparse
import pickle

from encoder_decoder import *
from data_prep import *

def evaluate(batches, encoder, decoder, char2i, use_cuda, loss=nn.NLLLoss(ignore_index=PADDING_index)):
    encoder.eval()
    decoder.eval()
    correct = 0
    total = 0

    for batch in batches:
        preds = predict(batch, encoder, decoder, char2i, use_cuda)
        targets = batch.output.t()
        target_lengths = batch.output_lengths



        for i, target in enumerate(targets):
            total += 1
            if target[:target_lengths[i]] == preds[:target_lengths[i]]
                correct += 1

    print("ACC: %.4f" % (correct = total))




def predict(batch, encoder, decoder, char2i, use_cuda):
    i2char = {i: w for w, i in char2i.items()}

    encoder_hidden = encoder.initHidden(batch.size, use_cuda)

    encoder_outputs, encoder_hidden = encoder(batch.input.t(), batch.input_lengths)
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    decoder_input = Variable(torch.LongTensor([EOS_index] * batch.size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # Use last (forward) hidden state from encoder
    decoder_hidden = encoder_hidden[:-1, :, :]

    all_decoder_outputs = Variable(torch.zeros(batch.max_length_out, batch.size, decoder.output_size))
    all_decoder_outputs = all_decoder_outputs.cuda() if use_cuda else all_decoder_outputs

    for t in range(batch.max_length_out):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs, use_cuda
            )

            topv, topi = decoder_output.data.topk(1)
            all_preds[t] = topi
            decoder_input = topi

    """
    input_variable = variableFromSentence(sentence, char2i, use_cuda)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden(use_cuda)

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        # Not sure why we sum them?
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[EOS_index]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                decoder_hidden, encoder_outputs, use_cuda)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_index:
                break
            else:
                decoded_words.append(i2char[ni])
                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    """
    return decoded_words, decoder_attentions[:di + 1]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Test the encoder decoder with reinflection data')
    parser.add_argument('filename', metavar='fn',
                        help='the filename of the file to train on')
    parser.add_argument('char2i', metavar='char2i', help='pkl file of the char2i dict from training')
    parser.add_argument('encoderModel', metavar='encoderModel',
                        help='The encoder that we are evaluating on')
    parser.add_argument('decoderModel', metavar='decoderModel',
                        help='The decoder that we are evaluating on')
    parser.add_argument('--gpu',action='store_true',  help='tell the system to use a gpu if you have cuda set up')

    args = parser.parse_args()
    hidden_size = 500
    test_data = DataPrep(args.filename)
    char2i_file = open(args.char2i, 'rb')

    train_char2i = pickle.load(char2i_file)
    encoder = torch.load(args.encoderModel)
    decoder = torch.load(args.decoderModel)

    use_cuda = args.gpu

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    acc = 0
    total = 0
    for isentence, osentence in test_data.pairs:
        pred, attentions = predict(encoder, decoder, isentence, train_char2i, use_cuda)
        print("===============")
        print("PREDICTION")
        print(pred)
        print("GOLD")
        print(osentence)
        if pred == osentence:
            acc  += 1
        total += 1

    print("ACCURACY: %2f" % (acc / total) * 100)
