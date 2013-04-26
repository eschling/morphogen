import sys, os, io
import argparse, logging
import uuid
import cPickle, gzip
import config
import tagset
from train import read_sentences

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

def get_attributes(tag):
    category = tagset.categories[tag[0]]
    for i, attr in enumerate(tag[1:], 1):
        if attr != '-':
            yield tagset.attributes[category, i]+'_'+attr

class Vocabulary(dict):
    def convert(self, feature):
        if feature not in self:
            self[feature] = len(self)
        return self[feature]

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Create cdec grammars')
    parser.add_argument('category', help='Russian word category to (R/V/A/N/M)')
    parser.add_argument('rev_map', help='reverse inflection map')
    parser.add_argument('output', help='training output path')
    args = parser.parse_args()

    logging.info('Loading reverse inflection map')
    with open(args.rev_map) as f:
        rev_map = cPickle.load(f)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    category = args.category

    clf_path = os.path.join(args.output, 'category_'+category)
    os.mkdir(clf_path)
    grammar_path = os.path.join(clf_path, 'grammars')
    os.mkdir(grammar_path)
    sgm = io.open(os.path.join(clf_path, 'train.sgm'), 'w', encoding='utf8')

    fvoc = Vocabulary()

    def expand_features(attr, features):
        for morph in get_attributes(category+attr):
            for fname, fval in features.iteritems():
                fid = fvoc.convert(u'{}_{}'.format(morph, fname))
                yield 'F{}={}'.format(fid, fval)

    n_sentences = 0
    logging.info('Generating grammars')
    for source, target, alignment in read_sentences(sys.stdin):
        n_sentences += 1
        if n_sentences % 100 == 0:
            sys.stderr.write('.')
        for word, features in extract_instances(category, source, target, alignment):
            inflection, lemma, tag = word
            category = tag[0]
            attributes = tag[1:]
            possible_inflections = rev_map.get((lemma, category), [])
            if (attributes, inflection) not in possible_inflections:
                logging.debug('Skip: %s (%s)', inflection, attributes)
                continue
            grammar_name = os.path.join(grammar_path, uuid.uuid1().hex+'.gz')
            src = lemma+'_'+category
            with gzip.open(grammar_name, 'w') as grammar:
                for (attr, _) in possible_inflections:
                    tgt = ' '.join(get_attributes(category+attr))
                    feat = ' '.join(expand_features(attr, features))
                    grammar.write(u'[S] ||| {} ||| {} {} ||| {}\n'.format(
                        src, category, tgt, feat).encode('utf8'))
            ref = ' '.join(get_attributes(tag))
            sgm.write(u'<seg grammar="{}"> {} ||| {} {} </seg>\n'.format(
                os.path.abspath(grammar_name), src, category, ref))

    logging.info('Saving weights')
    ff_path = os.path.join(args.output, 'category_'+category, 'weights.ini')
    with io.open(ff_path, 'w', encoding='utf8') as f:
        for fname, fid in fvoc.iteritems():
            f.write(u'# {}\n'.format(fname))
            f.write(u'F{} 0\n'.format(fid))

    sgm.close()

if __name__ == '__main__':
    main()
