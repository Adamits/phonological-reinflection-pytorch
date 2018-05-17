import argparse
import torch
import pickle
import codecs

from evaluate import predict, featurePredict
from data import *
from feature_data import DistinctiveFeatureData

EOS_symbol = "<EOS>"
EOS_index = 0
PAD_index = 1
PAD_symbol = "#"
UNK_index = 2
UNK_symbol='@'

def replace_UNK(input_text, input_encoded, pred):
    updated_pred = []
    if UNK_symbol in pred:
        print("UNKNOWN: %s:" % pred)
        in_copy = [i for i, c in enumerate(input_encoded)\
                   if c==UNK_symbol]

        copy_count = 0
        for p in pred:
            if p == UNK_symbol:
                # Replace the unknown with the corresponding
                # char in the input sequence.
                try:
                    """
                    Try to use the corresponding copy symbol.
                    """
                    updated_pred.append(input_text[in_copy[copy_count]])
                    copy_count+=1
                except:
                    """
                    Probably would make sense to just append most
                    frequent char here. Just randomly choose first char
                    of the input for now...
                    """
                    updated_pred.append(input_text[0])
            else:
                updated_pred.append(p)

        return ''.join(updated_pred)
    else:
        return pred

def replace_UNK_feature(input_text, input_encoded, pred, UNK_tensor):
    updated_pred = []
    if UNK_symbol in pred:
        in_copy = [i for i, c in enumerate(input_encoded)\
                   if c.equals(UNK_tensor)]

        copy_count = 0
        for p in pred:
            if p == UNK_symbol:
                # Replace the unknown with the correspinding
                # char in the input sequence.
                updated_pred.append(input_text[in_copy[copy_count]])
                copy_count+=1
            else:
                updated_pred.append(p)

        return ''.join(updated_pred)
    else:
        return pred

def make_predictions(pairs, encoder, decoder, char2i, outputfn,\
                     batch_size=100, use_cuda=True):
    i2char = {c: i for i, c in char2i.items()}
    batches = get_batches(pairs, batch_size, char2i,\
                          PAD_symbol, use_cuda, test_mode=True)
    output_strings = []
    out = open(outputfn, "w")
    for batch in batches:
        preds = predict(encoder, decoder, batch,\
                    list(char2i.keys()), use_cuda)

        for j in range(batch.size):
            eos = (preds[:, j] == EOS_index).\
                  nonzero().data[1][0]
            # Write the prediction up to the
            # second eos
            #print(''.join([i2char[int(c)] for c in\
            #               preds[1:eos, j]]))
            input_text = batch.inputs[j]
            input_enc = [i2char[int(c)] for c in \
                         batch.input_variable.t()[j]]
            pred = ''.join([i2char[int(c)] for c in\
                     preds[1:eos, j]])
            pred = replace_UNK(input_text, input_enc, pred)

            out.write(pred)
            out.write("\n")


def make_feature_predictions(pairs, encoder, decoder, char2i, outputfn,\
                    use_cuda=False, phonePairs=None):
    i2char = {c: i for i, c in char2i.items()}
    file_out = open(outputfn, "w")

    for p, data in enumerate(pairs):
        input_text, inp, output_text, out = data
        phone_in = None
        if phonePairs is not None:
            phone_in, phone_out = phonePairs[p]

        pred = ''.join([i2char[int(c)] for c in\
                    featurePredict(encoder, decoder, inp, use_cuda,\
                    phone_in=phone_in) if int(c) != EOS_index])
        
        # TODO When adding replace_UNK, probably need to account for whether EOS is there or not...

        UNK_feature_vector = torch.zeros(inp.size()[1])
        UNK_feature_vector[symbols2i[UNK_symbol]] = 1
        pred = replace_UNK_feature(input_text, inp, pred,\
                                   UNK_feature_vector)

        file_out.write(pred)
        file_out.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test")
    parser.add_argument('testfn', metavar='testfn',\
                        help='filename of test file')
    parser.add_argument('output_dir', metavar='output_dir',\
                        help='directory to write predictions to')
    parser.add_argument('encoder', metavar='encoder', \
                        help='The file for the encoder')
    parser.add_argument('decoder', metavar='decoder', \
                        help='The file for the decoder')
    parser.add_argument('char2i', metavar='char2i', \
                        help='Pickle file for the mapping')
    parser.add_argument('lang', metavar='lang',\
                        help='The language')
    parser.add_argument('setting', metavar='setting',\
                        help='low/medium/high')
    parser.add_argument('data_format',\
                        metavar='data_fomat',\
                        help='text, phone, or feature')
    parser.add_argument('batch_size', help=\
                        'The suize of each Batch')
    parser.add_argument('--phonechar2i', nargs='?')
    parser.add_argument('--symbols2i', nargs='?')
    parser.add_argument('--include_phone', action='store_true')
    parser.add_argument('--concat_phone', action='store_true')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    lang = args.lang
    setting = args.setting
    data_format = args.data_format
    include_phone = args.include_phone
    concat_phone = args.concat_phone    
    if data_format == "text":
        test_data = Data(args.testfn)
    elif data_format == "phone":
        test_data = PhoneData(args.testfn, args.lang)
    elif data_format == "feature":
        data_format = "feature_phone" if include_phone else data_format
        data_format = "feature_concat" if concat_phone  else data_format
        test_data = DistinctiveFeatureData(args.testfn, args.lang,\
                include_phone=include_phone, concat_phone=concat_phone)
        symbols2i = pickle.load(open(args.symbols2i, "rb"))

    encoder = torch.load(args.encoder)
    decoder = torch.load(args.decoder)
    char2i = pickle.load(open(args.char2i, "rb"))
    output_dir = args.output_dir
    output = "%s/%s-%s-preds-%s" % (output_dir, lang, data_format, setting)
    batch_size = int(args.batch_size)
    use_cuda = args.gpu

    print(char2i)
    if use_cuda:
        enocder = encoder.cuda()
        decoder = decoder.cuda()
    
    if data_format in ["feature", "feature_phone", "feature_concat"]:
        test_pairs = DistinctiveFeatureData.\
                     encode_examples(test_data.examples, symbols2i,\
                                     char2i, test_data.include_phone)
        phonePairs = None
        if concat_phone:
            phoneData = PhoneData(args.testfn, lang, segment_phone=True)
            phoneChar2i = pickle.load(open(args.phonechar2i, "rb"))
            phonePairs = phoneData.tensor_pairs(phoneChar2i)
            
        make_feature_predictions(test_pairs, encoder, decoder, char2i,\
                                 output, use_cuda, phonePairs=phonePairs)
    else:
        test_pairs = [([EOS_symbol] + i + [EOS_symbol],\
                       [EOS_symbol] + o + [EOS_symbol])\
                      for i, o in test_data.pairs]
        make_predictions(test_pairs, encoder, decoder, char2i, output,\
                         batch_size, use_cuda)
