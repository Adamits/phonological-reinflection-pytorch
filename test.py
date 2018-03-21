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
        # Flag so we only print one sample per batch
        neg_sample_printed = False
        for i in range(batch.size):
            preds = predict(batch, encoder, decoder, char2i, use_cuda, batch.mask_in)
            targets = batch.output.t()
            target_lengths = batch.output_lengths

            # sample = randint(0, batch.size - 1)
                    
            total += 1
            targets = targets.type(torch.cuda.FloatTensor) if use_cuda else targets.type(torch.FloatTensor)
            # Get the index of the second EOS to cut off pred at
            eos = (preds[:, i] == EOS_index).nonzero().data[1][0]
            
            if targets[:target_lengths[i], i].equal(preds[:eos+1, i]):
                correct += 1
            # Print one neg sample per batch
            elif neg_sample_printed == False:
                print("GOLD SAMPLE: ")
                print(''.join([i2char[int(c)] for c in targets[:target_lengths[i], i]]))
                print("BAD PRED: ")
                print(''.join([i2char[int(c)] for c in preds[:eos+1, i]]))
                print("=============================")
                neg_sample_printed = True

    return correct / total

def predict(batch, encoder, decoder, char2i, use_cuda, mask):
    #i2char = {i: w for w, i in char2i.items()}

    encoder_hidden = encoder.initHidden(batch.size, use_cuda)

    encoder_outputs, encoder_hidden = encoder(batch.input.t(), batch.input_lengths)
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    decoder_input = Variable(torch.LongTensor([EOS_index] * batch.size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # Use last (forward) hidden state from encoder
    decoder_hidden = decoder.initHidden(batch.size)

    all_preds = Variable(torch.zeros(batch.max_length_out, batch.size))
    all_preds = all_preds.cuda() if use_cuda else all_preds

    # Start at one in order to skip the first EOS, which we have already initialized
    for t in range(1, batch.max_length_out-1):
#        print("%i in TEST sequence!" % t)
#        print(i2char[decoder_input[0].data[0]])
#        print(batch.raw_in[0])
        # Add decoder_attn back
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs, use_cuda, mask
        )

        topv, topi = decoder_output.data.topk(1)
        topi = Variable(topi.squeeze(1))
        all_preds[t] = topi
        decoder_input = topi

    return all_preds

def testIter(encoder, decoder, pairs, char2i, use_cuda, batch_size=200):
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

    test_pairs = add_EOS_to_pairs(test_data.pairs)
    acc = testIter(encoder, decoder, test_pairs, char2i, use_cuda)
    print("ACC: %4f" % acc)
