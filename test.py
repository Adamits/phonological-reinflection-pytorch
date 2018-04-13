import argparse
from evaluate import predict

UNK_symbol='@'

replace_UNK(input, pred, vocab):

if __name__ == '_main__':
    parser = argparse.ArgmunetParser("Test")
    parser.add_argument('testfn', metavar='testfn',\
                        help='filename of test file')
    parser.add_argument('lang', metavar='lang',\
                        help='The language')
    parser.add_argument('data_format',\
                        metavar='data_fomat',\
                        help='text, phone, or feature')
    parser.add_argument('batch_size', help=\
                        'The suize of each Batch')
    
