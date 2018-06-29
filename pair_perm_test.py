import permutation_test as p
#help(p)

if __name__ == "__main__":
    phoneme_accs = [77, 82.3, 99.6, 82.8, 99.2, 88.2, 86.5, 96.4]
    feat_accs = [77.6, 85.3, 99.8, 82.9, 99.2, 89.2, 90.1, 96.2]
    p_value = p.functions.permutationtest(phoneme_accs, feat_accs)
