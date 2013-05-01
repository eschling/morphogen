import sys, os
import argparse, logging
import gzip, cPickle
import heapq, math, re
from itertools import izip
import config
from common import read_sentences
from crf_predict import CRFModel

sentence_re = re.compile('^<seg grammar="([^"]+)" id="(\d+)">(.+)</seg>$')
fields_re = re.compile('\s*\|\|\|\s*')

def read_sgm(fn):
    with open(fn) as f:
        for line in f:
            fields = fields_re.split(line[:-1])
            path, sid, src = sentence_re.match(fields[0]).groups()
            yield path, sid, src, fields[1:]

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Create synthetic phrases'
            ' using trained CRF models')
    parser.add_argument('lex_model', help='lexical translation model')
    parser.add_argument('rev_map', help='reverse inflection map')
    parser.add_argument('weights', nargs='+', help='trained models')
    parser.add_argument('sgm', help='original sentences + grammar pointers')
    parser.add_argument('out', help='grammar output directory')
    parser.add_argument('--threshold', type=float, default=0.01,
            help='lexical probability threshold')
    parser.add_argument('--ninfl', type=int, default=10,
            help='number of inflections per lemma')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    logging.info('Loading lexical translation model')
    tm = {}
    with gzip.open(args.lex_model) as f:
        for line in f:
            src, tgt, prob = line.decode('utf8').split()
            tm.setdefault(src, {})[tgt] = float(prob)

    logging.info('Loading reverse inflection map')
    with open(args.rev_map) as f:
        rev_map = cPickle.load(f)

    logging.info('Loading inflection prediction models')
    models = {}
    for fn in args.weights:
        category = fn[fn.find('.gz')-1]
        models[category] = CRFModel(fn)

    min_score = math.log(args.threshold)

    data = izip(read_sentences(sys.stdin), read_sgm(args.sgm))
    for (source, _, _), (grm_path, sid, left, right) in data:
        out_path = os.path.join(args.out, 'grammar.{}.gz'.format(sid))
        grammar_file = gzip.open(out_path, 'w')
        with gzip.open(grm_path) as f:
            for line in f:
                grammar_file.write(line)
        for j, src in enumerate(source):
            for tgt, lex_prob in tm.get(src.token, {}).iteritems():
                if lex_prob < min_score: continue
                lemma, category = tgt[:-2], tgt[-1]
                if category not in config.EXTRACTED_TAGS: continue
                possible_inflections = rev_map.get((lemma, category), [])
                if not possible_inflections: continue
                features = dict((fname, fval) for ff in config.FEATURES
                        for fname, fval in ff(source, lemma, j))
                model = models[category]
                scored_inflections = model.score_all(category,
                        possible_inflections, features)
                top_inflections = heapq.nlargest(args.ninfl, scored_inflections)
                for model_score, tag, inflection in top_inflections:
                    phrase_features = {
                        'Synthetic': 1,
                        'LexicalTranslation': lex_prob,
                        'Category_'+category: 1,
                        'InflectionScore': model_score
                    }
                    feat = ' '.join('{}={}'.format(*kv)
                            for kv in phrase_features.iteritems())
                    grammar_file.write(u'[X] ||| {} ||| {} ||| {} ||| 0-0\n'.format(
                        src.token, inflection, feat).encode('utf8'))
        grammar_file.close()
        new_left = '<seg grammar="{}" id="{}">{}</seg>'.format(out_path, sid, left)
        print(' ||| '.join([new_left] + right))

if __name__ == '__main__':
    main()
