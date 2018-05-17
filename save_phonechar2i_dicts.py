import argparse
import pickle
from data import PhoneData

data_format = "feature_concat"

if __name__=='__main__':
    parser = argparse.ArgumentParser("save those things")
    parser.add_argument('fn', metavar='fn')
    parser.add_argument('lang', metavar='lang')

    args = parser.parse_args()
    fn = args.fn
    lang = args.lang
    
    for setting in ["low", "medium"]:
        data=PhoneData(fn, lang, segment_phone=True)
        phoneChar2i = data.char2i
        
        phone_output =open('/home/adam/phonological-reinflection-pytorch/models/%s/phonechar2i-%s-%s.pkl' % (setting, lang, data_format), 'wb')
        pickle.dump(phoneChar2i, phone_output)
