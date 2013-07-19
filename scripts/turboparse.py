#!/usr/bin/env python
import sys
import os
from itertools import izip
import subprocess as sp
import multiprocessing as mp
import argparse

parser = None
def start_parser(parser_loc):
    turboparser = parser_loc+'/src/parser/TurboParser'
    #model_loc = parser_loc+'/models/sd_2_0_4_basic.model'
    model_loc = parser_loc+'/models/210full_sd204.model'
    global parser
    parser = sp.Popen([turboparser, '--test', '--file_model='+model_loc,
        '--file_test=/dev/stdin', '--file_prediction=/dev/stdout','--logtostderr'],
        stdin=sp.PIPE, stdout=sp.PIPE)

def parse(line):
    global parser
    sentence, tagged = line[:-1].split(' ||| ')
    words = sentence.split()
    tags = tagged.split()
    assert len(words) == len(tags)
    for i, (word, tag) in enumerate(izip(words, tags), 1):
        stag = tag if tag in ('PRP', 'PRP$') else tag[:2]
        parser.stdin.write('{}\t{}\t_\t{}\t{}\t_\t_\t_\n'.format(i, word, stag, tag))
    parser.stdin.write('\n')
    parser.stdin.flush()
    parse = []
    line = parser.stdout.readline()
    while line != '\n':
        _, _, _, _, _, _, head, mod = line.split()
        parse.append((head, mod))
        line = parser.stdout.readline()
    assert len(parse) == len(words)
    parsed = ' '.join('{}-{}'.format(head, mod) for head, mod in parse)
    return sentence, tagged, parsed

def main():
    arg_parser = argparse.ArgumentParser(description='Parse tagged sentences using TurboParser')
    arg_parser.add_argument('-p','--parser',help='location of TurboParser')
    arg_parser.add_argument('-j', '--jobs', type=int, default=1,
            help='number of instances of the parser to start')
    arg_parser.add_argument('-c', '--chunk', type=int, default=100,
            help='data chunk size')
    args = arg_parser.parse_args()

    os.environ['LD_LIBRARY_PATH'] = args.parser+'/deps/local/lib'
    pool = mp.Pool(processes=args.jobs, initializer=start_parser(args.parser))

    for sentence, tagged, parsed in pool.imap(parse, sys.stdin, chunksize=args.chunk):
        print('{} ||| {} ||| {}'.format(sentence, tagged, parsed))

if __name__ == '__main__':
    main()
