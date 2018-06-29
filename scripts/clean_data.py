from epitran import Epitran
from util import lang2ISO
import codecs
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser("Clean Data")
    parser.add_argument('fn', metavar='fn')
    parser.add_argument('lang', metavar='lang')

    args = parser.parse_args()
    fn = args.fn
    lang = args.lang
    
    with codecs.open(fn, "r", encoding='utf-8') as file:
        lines = [l.strip().split('\t') for l in\
                 file]

        iso = lang2ISO(lang)
        epi = Epitran(iso)
        for lemma, wf, tags in lines:
            if lemma.isdigit() or wf.isdigit():
                print("Digit! %s, %s" % (lemma, wf))
            elif epi.transliterate(lemma) and epi.transliterate(wf):
                pass
            else:
                print("Cannot transliterate! %s, %s" % (lemma, wf))
