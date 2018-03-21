import random
import argparse
import pickle
from masked_cross_entropy import masked_cross_entropy

from encoder_decoder import *
from test import evaluate
from data_prep import *

EOS="<EOS>"
EOS_index=0

def evaluate_dev(pairs, encoder, decoder, char2i, use_cuda, batch_size=100):
    batches = get_batches(pairs, batch_size, char2i, use_cuda)
    return evaluate(batches, encoder, decoder, char2i, use_cuda)

def train(batch, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, use_cuda, mask, i2char, teacher_forcing=True):
    """
    Compute the loss and make the parameter updates for a single sequence,
    where loss is the average of losses for each in the sequence
    """
    encoder_hidden = encoder.initHidden(batch.size, use_cuda)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    encoder_outputs, encoder_hidden = encoder(batch.input.t(), batch.input_lengths)
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    decoder_input = Variable(torch.LongTensor([EOS_index] * batch.size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = decoder.initHidden(batch_size)

    all_decoder_outputs = Variable(torch.zeros(batch.max_length_out, batch.size, decoder.output_size))
    all_decoder_outputs = all_decoder_outputs.cuda() if use_cuda else all_decoder_outputs
    
    if teacher_forcing:
        # Run through decoder one time step at a time
        # Start at one in order to skip the first EOS, which we have already initialized
        for t in range(1, batch.max_length_out):
            # Note mask is telling the attention mechanism which indices it should
            # not care be attending to.
            #print("character #%i of %i in train sequence!" % (t, batch.max_length_out))
            #print(i2char[decoder_input[0].data[0]])
            #print(batch.raw_in[0])
            #print("(%i chars)" % len(batch.raw_in[0]))
            # add attn arg back on left of =
            decoder_output, decoder_hidden  = decoder(
                decoder_input, decoder_hidden, encoder_outputs, use_cuda, mask
            )

            all_decoder_outputs[t] = decoder_output
            # Next input. Transpose to select along the column,
            # one from each batch at index t.
            decoder_input = batch.output.t()[t]

    else:
        for t in range(batch.max_length_out):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs, use_cuda
        )

            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi

            if ni == EOS:
                # Still add the EOS index, and then just pad the rest of the vector before breaking
                all_decoder_outputs[t] = decoder_output
                all_decoder_outputs = torch.cat(all_decoder_outputs, torch.LongTensor(batch.size, batch.max_length_out-t))
                break

            all_decoder_outputs[t] = decoder_output

    loss = masked_cross_entropy(
        # batch x seq x classes
        all_decoder_outputs.transpose(0, 1).contiguous(),
        # batch x seq
        batch.output,
        batch.output_lengths,
        use_cuda
    )

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]

def trainIters(encoder, decoder, pairs, dev_pairs, char2i, epochs, use_cuda, learning_rate=0.01, batch_size=50, teacher_forcing=True):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss(ignore_index=EOS_index)
    loss_function = loss_function.cuda() if use_cuda else loss_function
    i2char = {i: c for c, i in char2i.items()}

    pairs = pairs
    print("Preparing batches...")
    batches = get_batches(pairs, batch_size, char2i, use_cuda)
    
    for epoch in range(1, epochs + 1):
        print("EPOCH %i" % epoch)
        random.shuffle(batches)
        losses = []
        
        for batch in batches:
            loss = train(batch, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, use_cuda, batch.mask_in, i2char, teacher_forcing)

            losses.append(loss)

        print("LOSS: %.4f" % (sum(losses) / len(losses)))
        print(evaluate_dev(dev_pairs, encoder, decoder, char2i, use_cuda))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train the encoder decoder with reinflection data')
    parser.add_argument('filename', metavar='fn',
                        help='the filename of the file to train on')
    parser.add_argument('devfilename', metavar='devfn',
                        help='the filename of the dev file to evaluate on')
    parser.add_argument('lang', metavar='lang',
                        help='The language that we are training on')
    parser.add_argument('batch_size', metavar='batch_size',
                        help='size of each mini-batch')
    parser.add_argument('epochs', metavar='epochs',
                        help='number of epochs to train the model')
    parser.add_argument('lr', metavar='learning_rate', help='learning rate for the optimizers')
    parser.add_argument('--gpu',action='store_true',  help='tell the system to use a gpu if you have cuda set up')

    args = parser.parse_args()
    hidden_size = 300
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    lr = float(args.lr)
    data = DataPrep(args.filename)
    dev_data = DataPrep(args.devfilename)
    input_size = len(data.char2i.keys())
    encoder1 = EncoderRNN(input_size+1, hidden_size)
    attn_decoder1 = AttnDecoderRNN(hidden_size, input_size+1, dropout_p=0.1)

    char2i = data.char2i
    # Store the character dictionary for use in testing
    char_output = open('./models/%s-char2i.pkl' % args.lang, 'wb')
    pickle.dump(char2i, char_output)

    use_cuda = args.gpu

    if use_cuda:
        # This should put all parameters of the state of encoder and decoder
        # Onto the GPU.
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    # Wrap all sentences in EOS tags
    pairs = add_EOS_to_pairs(data.pairs)
    dev_pairs = add_EOS_to_pairs(dev_data.pairs)
    trainIters(encoder1, attn_decoder1, pairs, dev_pairs, char2i, epochs, use_cuda, learning_rate=lr, batch_size=batch_size)

    torch.save(encoder1, "./models/%s-encoder" % args.lang)
    torch.save(attn_decoder1, "./models/%s-decoder" % args.lang)
