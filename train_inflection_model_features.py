import random
import argparse
import pickle

from encoder import *
from decoder import *
from feature_data import *
from data import PhoneData
from evaluate import evaluate, featureEvaluate

def train(featPairs, dev_pairs, lang, setting, encoder, decoder, char2i,\
          loss_function, optimizer, data_format, batch_size, use_cuda,\
          epochs=20, lr=.01, clip=2, phonePairs=None, phoneDevPairs=None,\
          phoneChar2i=None):

    last_dev_acc = 0.0
    for i in range(epochs):
        print("EPOCH: %i" % i)

        if encoder.concat_phone:
            pairs = list(zip(featPairs, phonePairs))
        else:
            pairs = featPairs

        random.shuffle(pairs)
        all_losses = []
        for data in pairs:
            optimizer.zero_grad()
            if encoder.concat_phone:
                feat_pairs, phone_pairs = data
                _, inp, _, out = feat_pairs
                phone_inp, phone_out = phone_pairs
                enc_out, enc_hidden=encoder(inp, phone_inp)
            else:
                _, inp, _, out = data
                # Returns tensors with the batch dims
                enc_out, enc_hidden =\
                    encoder(inp)

            decoder_input = Variable(\
                    torch.LongTensor([EOS_index]))
            decoder_input = decoder_input.cuda()\
                    if use_cuda else decoder_input

            # Set hidden state to decoder's h0 of batch_size
            decoder_hidden = decoder.init_hidden()

            targets = out
            losses=[]

            for t in range(1, len(out)):
                decoder_output, decoder_hidden=\
                    decoder(decoder_input,\
                            decoder_hidden,\
                            enc_out, use_cuda)

                loss = loss_function(\
                        decoder_output.squeeze(0),\
                        targets[t])
                # Note reduce = True for loss_function, so we have a list
                # of all losses in
                # the minibatch. So we sum them, to be acounted for when
                # averaging the entire batch
                losses.append(loss.sum())

                # The next input is the next target (Teacher Forcing)
                # char in the sequence
                decoder_input = targets[t]

            # Get average loss by all loss values
            # / number of values discounting padding
            seq_loss = sum(losses) /len(losses)

            seq_loss.backward()

            all_losses.append(seq_loss.data[0])

            params = list(encoder.parameters())\
                     + list(decoder.parameters())
            # Gradient norm clipping for updates
            nn.utils.clip_grad_norm(params, clip)

            for p in params:
                p.data.add_(-lr, p.grad.data)

        print("LOSS: %4f" % (sum(all_losses)/ \
                             len(all_losses)))

        dev_acc = featureEvaluate(encoder, decoder, char2i,\
                        dev_pairs, use_cuda, phonePairs=phoneDevPairs,\
                        phoneChar2i=phoneChar2i)
        print("ACC: %.2f %% \n" % dev_acc)

        # Overwrite saved model if dev acc is higher
        if dev_acc > last_dev_acc:
            torch.save(encoder, "/home/adam/phonological-reinflection-pytorch/models/%s/encoder-%s-%s" % (setting, lang, data_format))
            torch.save(decoder, "/home/adam/phonological-reinflection-pytorch/models/%s/decoder-%s-%s" % (setting, lang, data_format))

        last_dev_acc = dev_acc
            
if __name__=='__main__':
    parser = argparse.ArgumentParser("Train")
    parser.add_argument('fn', metavar='fn',help=\
            'the name of the file with the train data')
    parser.add_argument('devfn', metavar='devfn',\
            help='the name of the file w/ dev data')
    parser.add_argument('lang', metavar='lang', help=\
                        'the name of the language')
    parser.add_argument('setting', metavar='setting', help=\
                        'low/medium/high')
    parser.add_argument('epochs', metavar='epochs', help=\
                        'number of epochs to train')
    parser.add_argument('lr', metavar='lr', help=\
                        'learning rate for the optimizer')
    parser.add_argument('clip', metavar='clip', help=\
                        'The gradient norm value at which to clip')
    parser.add_argument('--include_phone', action='store_true',\
                        help='add feature in feature vector for phone')
    parser.add_argument('--concat_phone', action='store_true',\
       help='make feature vectors from features and concat phone vector')
    parser.add_argument('--gpu', action='store_true',\
                        help='train on the gpu')

    args = parser.parse_args()
    lang = args.lang
    setting = args.setting
    epochs = int(args.epochs)
    lr = float(args.lr)
    clip = float(args.clip)
    include_phone = args.include_phone
    concat_phone = args.concat_phone
    use_cuda = args.gpu

    EOS_index = 0
    EOS_symbol = "<EOS>"
    PAD_index = 1
    PAD_symbol = "#"
    UNK_index = 2
    EMBEDDING_SIZE = 100
    HIDDEN_SIZE = 100
    phoneInputSize = None
    phoneChar2i=None
    phonePairs = None
    phoneDevPairs = None
    
    if include_phone:
        data_format = "feature_phone"
        data = DistinctiveFeatureData(args.fn, lang,\
                                      include_phone=True)
        dev_data = DistinctiveFeatureData(args.devfn, lang,\
                                          include_phone=True)
    elif concat_phone:
        data_format = "feature_concat"
        phoneData = PhoneData(args.fn, lang, segment_phone=True) 
        phoneDevData = PhoneData(args.devfn, lang, segment_phone=True)
        data = DistinctiveFeatureData(args.fn, lang,\
                                      segment_phone=True)
        dev_data = DistinctiveFeatureData(args.devfn, lang,\
                                          segment_phone=True)
        phoneChar2i = phoneData.char2i
        phonePairs = phoneData.tensor_pairs(phoneChar2i)
        phoneDevPairs = phoneDevData.tensor_pairs(phoneChar2i)
        phoneInputSize = len(phoneChar2i.keys())
    else:
        data_format = "feature"
        data = DistinctiveFeatureData(args.fn, lang)
        dev_data = DistinctiveFeatureData(args.devfn, lang)

    # This is for the output character space
    # Used for decoding
    char2i = data.char2i
    # Need to get so we can save it, and
    # Use it for testing later
    symbols2i = data.symbols2i

    print(char2i)
    # print(phoneChar2i)
    pairs = DistinctiveFeatureData.encode_examples(\
                            data.examples, symbols2i, char2i, include_phone=data.include_phone)
    dev_pairs = DistinctiveFeatureData.encode_examples(\
                            dev_data.examples, symbols2i, char2i, include_phone=dev_data.include_phone)

#    print(pairs)
#    print(phoneData.pairs)
#    print("====================")
    # +2 for ' ' and '-'
    if include_phone:
        input_size = len(data.feature_vocab + data.symbol_vocab) +\
                                      len(char2i.keys()) + 2 
    else:
        input_size = len(data.feature_vocab + data.symbol_vocab) + 2
        
    output_size = len(char2i.keys())

    char_output = open('/home/adam/phonological-reinflection-pytorch/models/%s/char2i-%s-%s.pkl' % (setting, lang, data_format), 'wb')
    pickle.dump(char2i, char_output)

    symbol_output = open('/home/adam/phonological-reinflection-pytorch/models/%s/symbols2i-%s-%s.pkl' % (setting, lang, data_format), 'wb')
    pickle.dump(symbols2i, symbol_output)

    if concat_phone:
        phone_output =open('/home/adam/phonological-reinflection-pytorch/models/%s/phonechar2i-%s-%s.pkl' % (setting, lang, data_format), 'wb')
        pickle.dump(phoneChar2i, phone_output)
        
    loss_func = nn.NLLLoss(ignore_index=PAD_index, reduce=False)

    encoder = PhonologyEncoder(input_size, EMBEDDING_SIZE,\
                            HIDDEN_SIZE, bidirectional=True,\
                            concat_phone=concat_phone,\
                               phone_input_size=phoneInputSize)
        
    decoder = PhoneDecoder(HIDDEN_SIZE, EMBEDDING_SIZE,\
                           output_size, bidirectional_input=True)
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        loss_func = loss_func.cuda()

    params = list(encoder.parameters()) +\
             list(decoder.parameters())
    optimizer = torch.optim.SGD(params, lr=lr)

    train(pairs, dev_pairs, lang, setting, encoder,\
          decoder, char2i, loss_func, optimizer, data_format, 1,\
          use_cuda, epochs=epochs, lr=lr, clip=clip,\
          phonePairs=phonePairs, phoneDevPairs=phoneDevPairs,\
          phoneChar2i=phoneChar2i)
