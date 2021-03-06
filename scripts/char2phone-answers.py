from data import PhoneData
import argparse
import codecs

if __name__=='__main__':
    parser = argparse.ArgumentParser("char2phone")
    parser.add_argument('fn')
    parser.add_argument('lang', metavar='lang')

    args = parser.parse_args()
    fn = args.fn
    lang = args.lang

    data = PhoneData(fn, lang)
    with codecs.open("./preds/%s-phone-answers" % (lang), "w") as out:
        out.write("\n".join([''.join(wf) for _, wf in data.pairs]))
