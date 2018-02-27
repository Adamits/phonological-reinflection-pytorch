import codecs

import torch
from torch.autograd import Variable

# Remember to add this to char2i
EOS="<EOS>"

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

class DataPrepPhones(DataPrep):
  def __init__():
    super(DataPrepPhones, self).__init__()

class DataPrepPhoneFeatures(DataPrep):
  def __init__():
    super(DataPrepPhoneFeatures, self).__init__()
