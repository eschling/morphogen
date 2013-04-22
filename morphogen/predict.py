import sys
import argparse
import logging
import cPickle
from train import FEATURES

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Show trained model')
    parser.add_argument('model', help='trained model')
    parser.add_argument('rev_map', help='reverse inflection map')
    args = parser.parse_args()

    logging.info('Loading inflection prediction model')
    with open(args.model) as f:
        m = cPickle.load(f)
    logging.info('Loaded model with %d categories', len(m))

    logging.info('Loading reverse inflection map')
    with open(args.rev_map) as f:
        rev_map = cPickle.load(f)

    for line in sys.stdin:
        source, lemma, category = line[:-1].decode('utf8').split(' ||| ')
        vectorizer, model = m[category]
        features = dict((fname, fval) for ff in FEATURES
                for fname, fval in ff(source, lemma, -1)) # TODO position
        fvector = vectorizer.transform(features)
        predictions = dict(zip(model.classes_, model.predict_proba(fvector)[0]))
        scored_inflections = ((predictions.get(tag, 0), inflection)
                for tag, inflection in rev_map[lemma, category])
        scored_inflections = sorted(scored_inflections, reverse=True)
        print(' '.join(u'{}/{:.3f}'.format(inflection, p)
            for p, inflection in scored_inflections if p > 1e-3).encode('utf8'))

if __name__ == '__main__':
    main()
