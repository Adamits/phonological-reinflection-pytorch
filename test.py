import argparse
import pickle
from random import randint

from encoder_decoder import *
from data_prep import *

def evaluate(batches, encoder, decoder, char2i, use_cuda):
    i2char = {i: w for w, i in char2i.items()}
    correct = 0
    total = 0

    for batch in batches:
        preds = predict(batch, encoder, decoder, char2i, use_cuda)
        targets = batch.output.t()
        target_lengths = batch.output_lengths

        sample = randint(0, batch.size - 1)
        
        for i in range(batch.size):
            total += 1
            targets = targets.type(torch.cuda.FloatTensor) if use_cuda else targets.type(torch.FloatTensor)

            if i == sample:
                print("GOLD SAMPLE: ")
                print([(i2char[int(c)], int(c)) for c in targets[:target_lengths[i], i]])
                print("PRED: ")
                print([(i2char[int(c)], int(c)) for c in preds[:target_lengths[i], i]])
                print("=============================")
            if targets[:target_lengths[i], i].equal(preds[:target_lengths[i], i]):
                correct += 1

    return correct / total

def predict(batch, encoder, decoder, char2i, use_cuda):
    #i2char = {i: w for w, i in char2i.items()}

    encoder_hidden = encoder.initHidden(batch.size, use_cuda)

    encoder_outputs, encoder_hidden = encoder(batch.input.t(), batch.input_lengths)
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    decoder_input = Variable(torch.LongTensor([EOS_index] * batch.size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # Use last (forward) hidden state from encoder
    decoder_hidden = encoder_hidden[:-1, :, :]

    all_preds = Variable(torch.zeros(batch.max_length_out, batch.size))
    all_preds = all_preds.cuda() if use_cuda else all_preds

    for t in range(batch.max_length_out):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs, use_cuda
            )

            topv, topi = decoder_output.data.topk(1)
            topi = Variable(topi.squeeze(1))
            all_preds[t] = topi
            decoder_input = topi

    return all_preds

def testIter(encoder, decoder, pairs, char2i, use_cuda, batch_size=100):
    print("Preparing batches...")
    batches = get_batches(pairs, batch_size, char2i, use_cuda)

    return evaluate(batches, encoder, decoder, char2i, use_cuda)

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

    char2i = pickle.load(char2i_file)
    encoder = torch.load(args.encoderModel)
    decoder = torch.load(args.decoderModel)

    use_cuda = args.gpu

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    else:
        encoder = encoder.cpu()
        decoder = decoder.cpu()

    acc = testIter(encoder, decoder, test_data.pairs, char2i, use_cuda)
    print("ACC: %4f" % acc)
