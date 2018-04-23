from data import *

import random
import pickle

def evaluate(encoder, decoder, char2i, pairs,\
             batch_size, PAD_symbol, use_cuda):
    correct = 0
    total = 0
    batches = get_batches(pairs, batch_size,\
                char2i, PAD_symbol, use_cuda)

    random.shuffle(batches)
    
    for i, batch in enumerate(batches):
        preds = predict(encoder, decoder,\
                batch, list(char2i.keys()),  use_cuda)
        # Squeeze off the batch_size = 1 dim
        targets = batch.output_variable.t()
        targets = targets.type(torch.cuda.FloatTensor)\
                  if use_cuda else \
                     targets.type(torch.FloatTensor)
        target_lengths = batch.lengths_out
        
        for j in range(batch.size):
            # Find index of second EOS
            eos = (preds[:, j] == EOS_index).\
                  nonzero().data[1][0]
            total+=1
            # Print at an arbitrary point in the data
            if False:
                print("=======")
                print("PREDS")
                print(preds[:eos+1,j])
                print("TARGETS")
                print(targets[:target_lengths[j],j])

            # Cut off pred at 2nd EOS
            equal = targets[:target_lengths[j], j]\
                    .equal(preds[:eos+1, j])
            if equal:
                correct += 1

    print("Accuracy: %.2f %% \n" % (correct / total * 100))

def predict(encoder, decoder, batch, vocab, use_cuda):
    enc_out, enc_hidden = encoder(\
                    batch.input_variable.t())

    decoder_input = Variable(torch.LongTensor(\
                    [EOS_index] * batch.size))
    decoder_input = decoder_input.cuda() if use_cuda\
                             else decoder_input

    dec_hidden = decoder.init_hidden(batch.size)

    all_preds = Variable(torch.zeros(batch.\
                        max_length_out, batch.size))
    all_preds = all_preds.cuda() if use_cuda else\
                all_preds

    for i in range(1, batch.max_length_out-1):
        dec_out, dec_hidden = decoder(decoder_input,\
            dec_hidden, enc_out, batch.size, use_cuda, batch.input_mask)

        topv, topi = dec_out.data.topk(1)
        topi = Variable(topi.squeeze(1))
        all_preds[i] = topi
        decoder_input = topi.view(batch.size)

    return all_preds

def featureEvaluate(encoder, decoder, char2i, pairs, use_cuda):
    correct = 0
    total = 0
    i2char = {i: c for c, i in char2i.items()}
    i=0
    print_at = random.sample(range(1, 1000), 5)
    for input_text, inp, output_text, out in pairs:
        preds = featurePredict(encoder, decoder, inp, use_cuda)
        targets = out
        total+=1
        if i in print_at:
            print("===============")
            print(input_text)
            print(''.join([i2char[int(c)] for c in preds]))
            print(output_text)
            print(''.join([i2char[int(c)] for c in targets]))
            print("===============")

        i+=1
        if targets.equal(preds):
            correct += 1

    return "Accuracy: %.2f %% \n" % (correct / total * 100)

def featurePredict(encoder, decoder, inp, use_cuda):
    enc_out, enc_hidden = encoder(inp)

    decoder_input = Variable(torch.LongTensor([EOS_index]))

    dec_hidden = decoder.init_hidden()

    preds = [EOS_index]
    for _ in range(enc_out.size()[0] * 3):
        dec_out, dec_hidden = decoder(decoder_input,\
                        dec_hidden, enc_out, use_cuda)

        topv, topi = dec_out.data.topk(1)
        # Get the index from the tensor as an integer
        topi_int = topi[0][0][0]

        preds.append(topi_int)
        decoder_input = Variable(topi.view(1))
        if topi_int == EOS_index:
            break

    return Variable(torch.LongTensor(preds))
                                                                        
if __name__=='__main__':
    char2i = pickle.load(open("./models/char2i.pkl", "rb"))
    label2i = pickle.load(open("./models/label2i.pkl", "rb"))
    encoder = torch.load("./models/encoder")
    acceptor = torch.load("./models/acceptor")

    langs = [("finnish", "/home/adam/phonological-reinflection-pytorch/data/finnish-uncovered-test"), \
             ("estonian", "/home/adam/phonological-reinflection-pytorch/data/estonian-uncovered-test")]

    all_pairs = []
    for lang, fn in langs:
        pairs, vocab = get_data(fn, lang)
        all_pairs += pairs

    pair_vars = [(wf2Var(wf, char2i), label2Var(l, label2i)) for wf, l in\
                 all_pairs]
    evaluate(encoder, decoder, pair_vars)
