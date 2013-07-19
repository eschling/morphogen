import sys
import argparse
import logging
from collections import defaultdict, Counter
import cPickle
import config_files.config
from common import read_sentences

def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Create reverse inflection map')
    parser.add_argument('-t','--tags',default='W',help='tags to produce mappings for')
    parser.add_argument('rev_map', help='output file')
    args = parser.parse_args()
    
    lemma_map = defaultdict(lambda: defaultdict(Counter))
    logging.info('Finding inflections and counting tag/form occurences...')
    for line in sys.stdin:
      if not line.strip(): continue #skip empty lines
      tgts, lemmas, tags = line.decode('utf8')[:-1].split(' ||| ')
      if not lemmas.strip(): continue #skip empty analyses
      target = zip(tgts.lower().split(), lemmas.lower().split(), tags.split())
      for (inflection, lemma, tag) in target:
          #logging.info('{} {} {}'.format(inflection, lemma, tag))
          if tag[0] not in args.tags: continue
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
