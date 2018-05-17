#!/usr/bin/env python
"""
Official Evaluation Script for the CoNLL-SIGMORPHON 2017 Shared Task.
Returns accuracy and mean Levenhstein distance.

Author: Ryan Cotterell
Last Update: 05/09/2017

Updated by Adam Wiemerslage for phone experiments
4/21/2018
"""

import numpy as np
import codecs

import epitran
from util import *

def distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    m = np.zeros([len(str2)+1, len(str1)+1])
    for x in range(1, len(str2) + 1):
        m[x][0] = m[x-1][0] + 1
    for y in range(1, len(str1) + 1):
        m[0][y] = m[0][y-1] + 1
    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            if str1[y-1] == str2[x-1]:
                dg = 0
            else:
                dg = 1
            m[x][y] = min(m[x-1][y] + 1, m[x][y-1] + 1, m[x-1][y-1] + dg)
    return int(m[len(str2)][len(str1)])

def read(fname):
    """ read file name """
    words = []
    with codecs.open(fname, 'rb', encoding='utf-8') as f:
        for line in f:
            words.append(line)

    return words

def eval_form(gold, guesses, lang, ignore=set()):
    """ compute average accuracy and edit distance for task 1 """
    epi = epitran.Epitran(lang2ISO(lang))

    text_guess, phone_guess, feature_guess = guesses
    #phone_guess, feature_guess = guesses
    correct, dist, total = 0., 0., 0.
    for i, gold_word in enumerate(gold):
        gold_word = gold_word.split("\t")[1]
        
        # Convert to phones
        gold_phone = get_phones(epi, gold_word)
            
        if gold_word == text_guess[i].strip() or gold_phone in [phone_guess[i].strip(), feature_guess[i].strip()]:
        #if gold_phone in [phone_guess[i].strip(), feature_guess[i].strip()]:
            correct += 1
        #else:
        #    print(gold_word, text_guess[i].strip(), gold_phone, phone_guess[i], feature_guess[i])
            
        #dist += distance(guess[i].strip(), gold_word)
        total += 1
        
    return (round(correct/total*100, 2), round(dist/total, 2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CoNLL-SIGMORPHON 2017 Shared Task Evaluation')
    parser.add_argument("--gold", help="Gold standard (uncovered)", required=True, type=str)
    parser.add_argument("--text_guess", help="Model output", required=True, type=str)
    parser.add_argument("--phone_guess", help="Model output", required=True, type=str)
    parser.add_argument("--feature_guess", help="Model output", required=True, type=str)
    parser.add_argument("--lang", help="Language being tested")

    args = parser.parse_args()

    gold = read(args.gold)
    guesses = [read(args.text_guess), read(args.phone_guess), read(args.feature_guess)]
    lang = args.lang

    print("{0:.2f}\t{1:.2f}".format(*eval_form(gold, guesses, lang)))
