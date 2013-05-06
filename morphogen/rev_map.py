import sys
import argparse
import logging
from collections import defaultdict, Counter
import cPickle
import config
from common import read_sentences

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Create reverse inflection map')
    parser.add_argument('rev_map', help='output file')
    args = parser.parse_args()
    
    lemma_map = defaultdict(lambda: defaultdict(Counter))
    logging.info('Finding inflections and counting tag/form occurences...')
    for _, target, _ in read_sentences(sys.stdin):
        for (inflection, lemma, tag) in target:
            if tag[0] not in config.EXTRACTED_TAGS: continue
            lemma_map[lemma, tag[0]][tag[1:]][inflection] += 1

    logging.info('Selecting most frequent form for each tag')
    rev_map = {lt: set() for lt in lemma_map.iterkeys()}
    for lt, inflections in lemma_map.iteritems():
        for tag, forms in inflections.iteritems():
            ((best_form, _),) = forms.most_common(1)
            rev_map[lt].add((tag, best_form))

    logging.info('Saving inflection map')
    with open(args.rev_map, 'w') as f:
        cPickle.dump(rev_map, f, protocol=-1)

if __name__ == '__main__':
    main()
