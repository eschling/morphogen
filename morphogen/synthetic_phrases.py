import sys, os
import argparse, logging
import gzip, cPickle
import heapq, math, numpy
from itertools import izip, groupby
from collections import namedtuple
import config
from common import read_sentences, read_sgm
from models import load_models

Candidate = namedtuple('Candidate', 'lex_score, rev_lex_score, inflection_score, '
                                    'inflection_rank, category, inflection')

def candidate_translations(rev_map, tm, rev_tm, crf_models, source, j):
    # For each possible lemma_category translation
    for tgt, lex_score in tm.get(source[j].token, {}).iteritems():
        if tgt not in rev_tm.get(source[j].token, {}): continue
        rev_lex_score = rev_tm[source[j].token][tgt]
        lemma, category = tgt[:-2], tgt[-1]
        # Find possible inflections of the lemma in this category
        if category not in config.EXTRACTED_TAGS: continue
        possible_inflections = rev_map.get((lemma, category), [])
        if not possible_inflections: continue
        # Score the inflections with the CRF models
        features = dict((fname, fval) for ff in config.FEATURES
                for fname, fval in ff(source, lemma, j))
        scored_inflections = crf_models[category].score_all(possible_inflections, features)
        # Marginalize over tags with same surface form
        ## 1. sort
        grouped_inflections = sorted([(score, inflection)
            for score, _, inflection in scored_inflections])
        ## 2. group by surface
        grouped_inflections = groupby(grouped_inflections, key=lambda t:t[1])
        ## 3. marginalize
        grouped_inflections = [(numpy.logaddexp.reduce([score for score, _ in group]),
            inflection) for inflection, group in grouped_inflections]
        # Compute ranks (sort by decreasing score)
        sorted_inflections = sorted(grouped_inflections, reverse=True)
        # Produce candidate translations
        for rank, (inflection_score, inflection) in enumerate(sorted_inflections):
            yield Candidate(lex_score, rev_lex_score, inflection_score, rank,
                    category, inflection)

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Create synthetic phrases'
            ' using trained CRF models')
    parser.add_argument('lex_model', help='lexical translation model')
    parser.add_argument('rev_lex_model', help='reverse lexical translation model')
    parser.add_argument('rev_map', help='reverse inflection map')
    parser.add_argument('models', nargs='+', help='trained models')
    parser.add_argument('sgm', help='original sentences + grammar pointers')
    parser.add_argument('out', help='grammar output directory')
    parser.add_argument('-t', '--threshold', type=float, default=0.01,
            help='lexical probability threshold')
    parser.add_argument('-n', '--n_candidates', type=int, default=30,
            help='number of inflections per lemma')
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.mkdir(args.out)

    min_score = math.log(args.threshold)
    logging.info('Loading lexical translation models')
    tm = {}
    with gzip.open(args.lex_model) as f:
        for line in f:
            src, tgt, prob = line.decode('utf8').split()
            prob = float(prob)
            if prob > min_score:
                tm.setdefault(src, {})[tgt] = prob
    rev_tm = {}
    with gzip.open(args.rev_lex_model) as f:
        for line in f:
            src, tgt, prob = line.decode('utf8').split()
            rev_tm.setdefault(tgt, {})[src] = float(prob)

    logging.info('Loading reverse inflection map')
    with open(args.rev_map) as f:
        rev_map = cPickle.load(f)

    logging.info('Loading inflection prediction models')
    models = load_models(args.models)

    logging.info('Generating extended grammars')
    data = izip(read_sentences(sys.stdin, skip_empty=False), read_sgm(args.sgm))
    for (source, _, _), (grm_path, sid, left, right) in data:
        # Create grammar
        out_path = os.path.join(args.out, 'grammar.{}.gz'.format(sid))
        grammar_file = gzip.open(out_path, 'w')
        # Copy original grammar
        with gzip.open(grm_path) as f:
            for line in f:
                grammar_file.write(line)
        # Generate synthetic phrases
        for j, src in enumerate(source):
            candidates = candidate_translations(rev_map, tm, rev_tm, models, source, j)
            top_candidates = heapq.nlargest(args.n_candidates, candidates)
            for candidate in top_candidates:
                phrase_features = {
                    'Synthetic': 1,
                    'LexicalTranslation': candidate.lex_score,
                    'ReverseLexicalTranslation': candidate.rev_lex_score,
                    'Category_'+candidate.category: 1,
                    'InflectionScore': candidate.inflection_score,
                    'InflectionRank': candidate.inflection_rank
                }
                feat = ' '.join('{}={}'.format(*kv)
                        for kv in phrase_features.iteritems())
                grammar_file.write(u'[X] ||| {} ||| {} ||| {} ||| 0-0\n'.format(
                    src.token, candidate.inflection, feat).encode('utf8'))
        grammar_file.close()
        # Write sgm
        new_left = u'<seg grammar="{}" id="{}">{}</seg>'.format(out_path, sid, left)
        print(u' ||| '.join([new_left] + right).encode('utf8'))

if __name__ == '__main__':
    main()
