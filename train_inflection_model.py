import random
import argparse
import pickle

from encoder import *
from decoder import *
from data import *
from evaluate import evaluate

def train(pairs, dev_pairs, lang, encoder, acceptor, loss_function, optimizer, use_cuda, batch_size=100, epochs=20, lr=.01, clip=2):
    random.shuffle(pairs)
    train_batches = get_batches(pairs, batch_size,\
                char2i, PAD_symbol, use_cuda)
    for i in range(epochs):
        print("EPOCH: %i" % i)
        random.shuffle(train_batches)

        all_losses = []
        for batch in train_batches:
            optimizer.zero_grad()

            # Returns tensors with the batch dims
            enc_out, enc_hidden =\
                    encoder(batch.input_variable.t())

            decoder_input = Variable(\
                    torch.LongTensor([EOS_index] *\
                    batch.size))
            decoder_input = decoder_input.cuda()\
                    if use_cuda else decoder_input

            # Set hidden state to decoder's h0 of batch_size
            decoder_hidden = decoder.init_hidden(batch_size)

            targets = batch.output_variable.t()
            losses=[]

            for t in range(1, batch.max_length_out):
                decoder_output, decoder_hidden=\
                    decoder(decoder_input,\
                            decoder_hidden,\
                            enc_out, batch.size,\
                            use_cuda, batch.input_mask)

                # Find the loss for a single character, to be averaged
                # over all non-padding predictions. Squeeze the batch\
                # dim =1 off of the dec output
                loss = loss_function(\
                        decoder_output.squeeze(0),\
                        targets[t])
                # Note reduce = True for loss_function, so we have a list
                # of all losses in
                # the minibatch. So we sum them, to be acounted for when
                # averaging the entire batch
                losses.append(loss.sum())

                # The next input is the next target
                # char in the sequence
                decoder_input = batch.output_variable.t()[t]

            # Get average loss by all loss values
            # / number of values discounting padding
            seq_loss = sum(losses) / sum(batch.lengths_out)

            seq_loss.backward()

            all_losses.append(seq_loss)

            # Gradient norm clipping for updates
            nn.utils.clip_grad_norm(list(encoder.parameters()) + list(decoder.parameters()), clip)
            for p in list(encoder.parameters()) + list(decoder.parameters()):
                p.data.add_(-lr, p.grad.data)

            # optimizer.step()
                
        print("LOSS: %4f" % (sum(all_losses)/ \
                             len(all_losses)))

        evaluate(encoder, decoder, char2i, dev_pairs,\
                 batch_size, PAD_symbol, use_cuda)
        
        torch.save(encoder, "./models/encoder-%s" % lang)
        torch.save(acceptor, "./models/acceptor-%s" % lang)

if __name__=='__main__':
    parser = argparse.ArgumentParser("Train")
    parser.add_argument('fn', metavar='fn',help=\
            'the name of the file with the train data')
    parser.add_argument('devfn', metavar='devfn',\
            help='the name of the file w/ dev data')
    parser.add_argument('lang', metavar='lang', help=\
                        'the name of the language')
    parser.add_argument('data_format', metavar='data_format',\
                        help='text, phone, or feature')
    parser.add_argument('epochs', metavar='epochs', help=\
                        'number of epochs to train')
    parser.add_argument('batch_size', metavar='bs', help=\
                        'the size of each mini-batch')
    parser.add_argument('lr', metavar='lr', help=\
                        'learning rate for the optimizer')
    parser.add_argument('clip', metavar='clip', help=\
                        'The gradient norm value at which to clip')
    parser.add_argument('--gpu', action='store_true',\
                        help='train on the gpu')

    args = parser.parse_args()
    lang = args.lang
    data_format = args.data_format
    if data_format == "text":
        data = Data(args.fn)
        dev_data = Data(args.devfn)
    elif data_format == "phone":
        data = PhoneData(args.fn, lang)
        dev_data = PhoneData(args.fn, lang)
    elif data_format == "feature":
        raise Exception("unimplemented feature")
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    clip = float(args.clip)
    use_cuda = args.gpu
    
    EOS_index = 0
    EOS_symbol = "<EOS>"
    PAD_index = 1
    PAD_symbol = "#"
    UNK_index = 2
    EMBEDDING_SIZE = 100
    HIDDEN_SIZE = 100

    char2i = data.char2i
    input_size = output_size = len(char2i.keys())

    char_output = open('./models/char2i-%s.pkl' % lang, 'wb')
    pickle.dump(char2i, char_output)
    
    loss_func = nn.NLLLoss(ignore_index=PAD_index, reduce=False)
    encoder = Encoder(input_size, EMBEDDING_SIZE,\
                      HIDDEN_SIZE, bidirectional=True)
    decoder = Decoder(HIDDEN_SIZE, EMBEDDING_SIZE,\
                      output_size, bidirectional_input=True)

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        loss_func = loss_func.cuda()

    params = list(encoder.parameters()) +\
             list(decoder.parameters())
    optimizer = torch.optim.SGD(params, lr=lr)

    # Wrap both sets in EOS
    pairs = [([EOS_symbol] + i + [EOS_symbol],\
              [EOS_symbol] + o + [EOS_symbol])\
             for i, o in data.pairs]
    dev_pairs = [([EOS_symbol] + i + [EOS_symbol],\
                  [EOS_symbol] + o + [EOS_symbol])\
                 for i, o in dev_data.pairs]
    
    train(pairs, dev_pairs, lang, encoder,\
          decoder, loss_func, optimizer, use_cuda,\
          batch_size=batch_size, epochs=epochs, lr=lr, clip=clip)
