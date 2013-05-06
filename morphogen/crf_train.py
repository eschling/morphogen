import sys, os, io
import uuid
import argparse, logging
import cPickle
import config
import tagset
from common import read_sentences

def extract_instances(category, source, target, alignment):
    """Extract (category, features, tag) training instances for a sentence pair"""
    for i, (token, lemma, tag) in enumerate(target):
        if tag[0] != category: continue
        word_alignments = [j for (k, j) in alignment if k == i] # tgt == i - src
        if len(word_alignments) != 1: continue # Extract only one-to-one alignments
        (j,) = word_alignments # src
        features = dict((fname, fval) for ff in config.FEATURES
                for fname, fval in ff(source, lemma, j))
        yield (token, lemma, tag), features

def get_attributes(cat, attrs):
    category = tagset.categories[cat]
    for i, attr in enumerate(attrs, 1):
        if attr != '-':
            yield tagset.attributes[category, i]+'_'+attr

class Vocabulary(dict):
    def convert(self, feature):
        if feature not in self:
            self[feature] = len(self)
        return self[feature]

    def expand_features(self, category, attributes, features):
        for morph in get_attributes(category, attributes):
            # target features
            fid = self.convert(morph)
            yield 'F{}=1'.format(fid)
            # pairwise features
            for morph2 in get_attributes(category, attributes):
                if morph2 <= morph: continue
                fid = self.convert(u'{}+{}'.format(morph, morph2))
                yield 'F{}=1'.format(fid)
            # translation features
            for fname, fval in features.iteritems():
                fid = self.convert(u'{}_{}'.format(morph, fname))
                yield 'F{}={}'.format(fid, fval)

    def make_rule(self, lemma, category, attributes, features):
        src = lemma+'_'+category
        tgt = ' '.join(get_attributes(category, attributes))
        feat = ' '.join(self.expand_features(category, attributes, features))
        return (u'[S] ||| {} ||| {} {} ||| {}\n'.format(src, category, tgt, feat))

import subprocess as sp
def too_much_mem():
    p = sp.Popen(('df', '/dev/shm'), stdout=sp.PIPE)
    out, _  = p.communicate()
    percent = float(out.split()[9]) / float(out.split()[8])
    return (percent > 0.9)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Create cdec CRF grammars and training data')
    parser.add_argument('category', help='Russian word category to (R/V/A/N/M)')
    parser.add_argument('rev_map', help='reverse inflection map')
    parser.add_argument('output', help='training output path')
    args = parser.parse_args()

    category = args.category

    logging.info('Loading reverse inflection map')
    with open(args.rev_map) as f:
        rev_map = cPickle.load(f)

    # Create training data paths
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    grammar_path = os.path.join(args.output, 'grammars')
    if not os.path.exists(grammar_path):
        os.mkdir(grammar_path)

    sgm = io.open(os.path.join(args.output, 'train.sgm'), 'w', encoding='utf8')

    fvoc = Vocabulary()

    n_sentences = 0
    logging.info('Generating the grammars')
    for source, target, alignment in read_sentences(sys.stdin):
        n_sentences += 1
        if n_sentences % 1000 == 0:
            if too_much_mem():
                logging.info('Running out of memory')
                break
        for word, features in extract_instances(category, source, target, alignment):
            inflection, lemma, tag = word
            category = tag[0]
            ref_attributes = tag[1:]
            possible_inflections = rev_map.get((lemma, category), [])
            if (ref_attributes, inflection) not in possible_inflections:
                logging.debug('Skip: %s (%s)', inflection, ref_attributes)
                continue
            # Write sentence grammar
            grammar_name = os.path.join(grammar_path, uuid.uuid1().hex)
            with io.open(grammar_name, 'w', encoding='utf8') as grammar:
                for attributes, _ in possible_inflections:
                    rule = fvoc.make_rule(lemma, category, attributes, features)
                    grammar.write(rule)
            # Write src / ref
            src = lemma+'_'+category
            ref = ' '.join(get_attributes(category, ref_attributes))
            sgm.write(u'<seg grammar="{}"> {} ||| {} {} </seg>\n'.format(
                os.path.abspath(grammar_name), src, category, ref))

    logging.info('Processed %d sentences', n_sentences)
    logging.info('Saving weights')
    ff_path = os.path.join(args.output, 'weights.ini')
    with io.open(ff_path, 'w', encoding='utf8') as f:
        for fname, fid in fvoc.iteritems():
            f.write(u'# {}\n'.format(fname))
            f.write(u'F{} 0\n'.format(fid))

    sgm.close()

if __name__ == '__main__':
    main()
