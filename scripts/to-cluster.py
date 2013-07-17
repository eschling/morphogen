#!/usr/bin/env python
import sys
import logging
import argparse
import gzip
import pickle

def main():
    parser = argparse.ArgumentParser(description='Brown cluster grammar maker')
    parser.add_argument('clusters', help='cluster file')
    parser.add_argument('--entropy', help='optional entropy file')
    parser.add_argument('--cutoff', type=float, default=100, help='entropy cutoff')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cluster = {}

    logging.info('Reading clusters...')
    with gzip.open(args.clusters) as f:
        for line in f:
            c, word, count = line.decode('utf8').split('\t')
            cluster[word] = c

    if args.entropy:
        with open(args.entropy) as f:
            entropy = pickle.load(f)
        def convert(word):
            c = cluster.get(word)
            if not c or entropy[c] < args.cutoff: return word
            return 'C'+c
    else:
        def convert(word):
            return 'C'+cluster.get(word, 'UNK')

    for line in sys.stdin:
        print(' '.join(map(convert, line.decode('utf8').split())).encode('utf8'))

if __name__ == '__main__':
    main()
