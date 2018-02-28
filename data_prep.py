import codecs

import torch
from torch.autograd import Variable

# Remember to add this to char2i
EOS="<EOS>"
EOS_index=0

class DataPrep:
  def __init__():
    test = True

  def readData(fn):
      print("Reading lines...")

      # Read the file and split into lines
      with codecs.open(fn, "r", encoding='utf-8') as file:
        lines = [l.strip().split('\t') for l in file]

      # Split every line into pairs
      # Form of [[e, a, c, h, " ", c, h, a, r, tag, tag, tag], w, o, r, d, " ", f, o, r, m]
      pairs = [(list(lemma) + tags.split(";"), list(wf)) for lemma, wf, tags in lines]

      return pairs

  def prepareData(fn):
      pairs = readData(fn)
      print("Counting chars...")
      for pair in pairs:
          inp, outp = pair
          input_data.append(inp)
          input_vocab.append([c for c in inp if inp not in input_vocab])
          output_data.append(outp)
          output_vocab.append([c for c in inp if outp not in output_vocab])


      return input_data, output_data, input_vocab, output_vocab

  def indexesFromSentence(sentence, char2i):
      return [char2i[c] for c in sentence]

  def variableFromSentence(sentence, char2i):
      indexes = indexesFromSentence(sentence, char2i)
      indexes.append(char2int[EOS])
      result = Variable(torch.LongTensor(indexes).view(-1, 1))
      if use_cuda:
          return result.cuda()
      else:
          return result

  def variablesFromPair(pair, char2i):
      input_variable = variableFromSentence(pair[0], char2i)
      target_variable = variableFromSentence(pair[1], char2i)
      return (input_variable, target_variable)

  def evaluate(encoder, decoder, sentence, max_length):
        input_variable = variableFromSentence(inp, sentence)
        input_length = input_variable.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

            decoder_input = Variable(torch.LongTensor([[EOS_index]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
              decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)
              decoder_attentions[di] = decoder_attention.data
              topv, topi = decoder_output.data.topk(1)
              ni = topi[0][0]
              if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
              else:
                decoded_words.append(outp.index2word[ni])

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input


        return decoded_words, decoder_attentions[:di + 1]

class DataPrepPhones(DataPrep):
  def __init__():
    super(DataPrepPhones, self).__init__()

class DataPrepPhoneFeatures(DataPrep):
  def __init__():
    super(DataPrepPhoneFeatures, self).__init__()

