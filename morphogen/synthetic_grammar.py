import sys, os, re
import argparse, logging
import gzip, cPickle
from itertools import izip
from collections import namedtuple, Counter
import config
from common import read_sentences, read_sgm
from models import load_models

_Rule = namedtuple('_Rule', 'lhs, rhs, features, alignment')
class Rule(_Rule):
    def __unicode__(self):
        fv = ' '.join('{}={}'.format(*kv) for kv in self.features.iteritems())
        al = ' '.join('{}-{}'.format(*ij) for ij in self.alignment)
        lhs = u' '.join(self.lhs)
        rhs = u' '.join(self.rhs)
        return u'[X] ||| {} ||| {} ||| {} ||| {}'.format(lhs, rhs, fv, al)

def read_grammar(fn):
    """Read a set of rules from a grammar"""
    with gzip.open(fn) as f:
        for line in f:
            _, lhs, rhs, features, alignment = line.decode('utf8').split(' ||| ')
            features = {k: float(v) for k, v in
                    (kv.split('=') for kv in features.split())}
            alignment = [(int(i), int(j)) for i, j in
                    (point.split('-') for point in alignment.split())]
            yield Rule(lhs.split(), rhs.split(), features, alignment)

def source_match(lhs, source):
    """Find all positions which match a phrase in the source sentence"""
    for j in xrange(len(source) - len(lhs) + 1):
        if all(x == y.token for x, y in zip(lhs, source[j:])):
            yield j

lemma_re = re.compile('^(.+)_(['+config.EXTRACTED_TAGS+'])$')
def synthetic_rule(rev_map, models, rule, source, match):
    """Create inflected rule"""
    rule_features = Counter(rule.features)
    rule_features['Synthetic'] = 1
    inflected_rhs = []
    for rule_j, tgt in enumerate(rule.rhs):
        # Skip non-predicted categories
        m = lemma_re.match(tgt)
        if not m:
            inflected_rhs.append(tgt)
            continue
        # Inflect predicted categories
        lemma, category = m.groups()
        possible_inflections = rev_map.get((lemma, category), [])
        if not possible_inflections: return
        # Translate alignment point
        alignments = [rule_i for rule_i, rule_k in rule.alignment if rule_k == rule_j]
        if len(alignments) != 1: return # skip non 1-n alignment
        i = match + alignments[0]
        assert 0 <= i < len(source)
        #logging.debug('Match for `%s`/%s: %s', unicode(rule), tgt, source[j].token)
        # Score the inflections with the CRF models
        inflection_features = dict((fname, fval) for ff in config.FEATURES
                for fname, fval in ff(source, lemma, i))
        scored_inflections = models[category].score_all(possible_inflections,
                inflection_features)
        # could produce multiple inflections here
        score, tag, inflection = max(scored_inflections)
        inflected_rhs.append(inflection)
        rule_features['Category_'+category] += 1
        rule_features['InflectionScore'] += score

    # FIXME Should we not create a rule if nothing is inflected?

    assert len(inflected_rhs) == len(rule.rhs)
    yield Rule(rule.lhs, inflected_rhs, rule_features, rule.alignment)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Create synthetic phrases'
            ' using trained CRF models and lemma grammar')
    parser.add_argument('rev_map', help='reverse inflection map')
    parser.add_argument('models', nargs='+', help='trained models (category:file)')
    parser.add_argument('sgm', help='original sentences + grammar pointers')
    parser.add_argument('sgm_lem', help='original sentences + lemma grammar pointers')
    parser.add_argument('out', help='grammar output directory')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    logging.info('Loading reverse inflection map')
    with open(args.rev_map) as f:
        rev_map = cPickle.load(f)

    logging.info('Loading inflection prediction models')
    models = load_models(args.models)

    logging.info('Generating extended grammars')
    data = izip(read_sentences(sys.stdin, skip_empty=False),
            read_sgm(args.sgm), read_sgm(args.sgm_lem))
    for (source, _, _), (grm_path, sid, left, right),\
            (lem_grm_path, lem_sid, lem_left, lem_right) in data:
        assert sid == lem_sid and left == lem_left and right == lem_right
        # Create grammar file
        out_path = os.path.join(args.out, 'grammar.{}.gz'.format(sid))
        grammar_file = gzip.open(out_path, 'w')
        # Copy original grammar
        with gzip.open(grm_path) as f:
            for line in f:
                grammar_file.write(line)

        # Generate synthetic phrases from lemma grammar
        for rule in read_grammar(lem_grm_path):
            assert not any(src.startswith('[X,') for src in rule.lhs) # no gaps, please
            for match in source_match(rule.lhs, source):
                # create (at most) a synthetic rule
                for new_rule in synthetic_rule(rev_map, models, rule, source, match):
                    grammar_file.write(unicode(new_rule).encode('utf8')+'\n')

        grammar_file.close()
        # Write sgm
        new_left = u'<seg grammar="{}" id="{}">{}</seg>'.format(out_path, sid, left)
        print(u' ||| '.join([new_left] + right).encode('utf8'))

if __name__ == '__main__':
    main()
