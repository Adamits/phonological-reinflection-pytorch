import random
import argparse
import pickle
from masked_cross_entropy import masked_cross_entropy

from encoder_decoder import *
from data_prep import *

EOS="<EOS>"
EOS_index=0
PADDING_SYMBOL = "@"
PADDING_index=0

def train(batch, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, use_cuda, teacher_forcing = True):
    """
    Compute the loss and make the parameter updates for a single sequence,
    where loss is the average of losses for each in the sequence
    """
    encoder_hidden = encoder.initHidden(batch.size, use_cuda)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    print("Encoding...")
    encoder_outputs, encoder_hidden = encoder(batch.input.t(), batch.input_lengths)
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    decoder_input = Variable(torch.LongTensor([EOS_index] * batch.size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # Use last (forward) hidden state from encoder
    decoder_hidden = encoder_hidden[:-1, :, :]

    all_decoder_outputs = Variable(torch.zeros(batch.max_length_out, batch.size, decoder.output_size))
    all_decoder_outputs = all_decoder_outputs.cuda() if use_cuda else all_decoder_outputs

    print("Decoding...")
    if teacher_forcing:
            # Run through decoder one time step at a time
            for t in range(batch.max_length_out):
                decoder_output, decoder_hidden, decoder_attn = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, use_cuda
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
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            if ni == EOS_token:
                # Still add the EOS index, and then just pad the rest of the vector before breaking
                all_decoder_outputs[t] = decoder_output
                all_decoder_outputs = torch.cat(all_decoder_outputs, torch.LongTensor(batch.size, batch.max_length_out-t))
                break

            all_decoder_outputs[t] = decoder_output

    print("Computing loss...")
    loss = masked_cross_entropy(
        # batch x seq x classes
        all_decoder_outputs.transpose(0, 1).contiguous(),
        # batch x seq
        batch.output,
        batch.output_lengths,
        use_cuda
    )

    print("Backpropogating loss...")
    loss.backward()

    print("Updating parameters...")
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]

def trainIters(encoder, decoder, pairs, char2i, epochs, use_cuda, learning_rate=0.01, batch_size=5, teacher_forcing=True):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss(ignore_index=PADDING_index)
    loss_function = loss_function.cuda() if use_cuda else loss_function
    print("Preparing batches...")
    sorted_pairs = pairs.copy()
    # Sort the data by the length of the output so that batches have similar lengths
    # We will perform more computations over output than input, presumably
    sorted_pairs.sort(key=lambda x: len(x[1]), reverse=True)

    # Split sorted_data into n batches each of size batch_length
    batches = [sorted_pairs[i:i+batch_size] for i in range(0, len(sorted_pairs), batch_size)]
    # Loop over indices so we can modify batches in place
    for i in range(len(batches)):
        batches[i] = Batch(batches[i])
        batches[i].input_variable(char2i, use_cuda)
        batches[i].output_variable(char2i, use_cuda)

    for epoch in range(1, epochs + 1):
        print("EPOCH %i" % epoch)
        random.shuffle(batches)
        losses = []

        for i, batch in enumerate(batches):
            print("computing batch %i" % i)

            loss = train(batch, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, use_cuda, teacher_forcing)
            print("Batch loss: %.4f" % loss)
            losses.append(loss)

        print("LOSS: %.4f" % (sum(losses) / len(losses)))

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
    attn_decoder1 = AttnDecoderRNN(hidden_size, input_size+1, dropout_p=0.3)

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

    trainIters(encoder1, attn_decoder1, data.pairs, char2i, 50, use_cuda, batch_size=400)

    torch.save(encoder1, "./models/%s-encoder" % args.lang)
    torch.save(attn_decoder1, "./models/%s-decoder" % args.lang)
