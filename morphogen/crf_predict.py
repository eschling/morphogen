import sys
import argparse, logging
import cPickle, gzip
import math, numpy
import config
from common import read_sentences
from crf_train import get_attributes
from predict import extract_instances

class CRFModel:
    def __init__(self, category, fn):
        self.category = category
        self.weights = {}
        with gzip.open(fn) as f:
            for line in f:
                fname, fval = line.decode('utf8').split()
                self.weights[fname] = float(fval)

    def score(self, tag, features):
        score = 0
        for attr in get_attributes(self.category, tag):
            for fname, fval in features.iteritems():
                score += fval * self.weights.get(attr+'_'+fname, 0)
        return score

    def score_all(self, inflections, features):
        scored = [(self.score(tag, features), tag, inflection)
                for tag, inflection in inflections]
        z = numpy.logaddexp.reduce([score for score, _, _ in scored])
        return [(score - z, tag, inflection) for score, tag, inflection in scored]

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Predict using trained CRF models')
    parser.add_argument('rev_map', help='reverse inflection map')
    parser.add_argument('weights', nargs='+', help='trained models')
    args = parser.parse_args()

    logging.info('Loading reverse inflection map')
    with open(args.rev_map) as f:
        rev_map = cPickle.load(f)

    models = {}
    logging.info('Loading inflection prediction models')
    for fn in args.weights:
        category = fn[fn.find('.gz')-1]
        models[category] = CRFModel(category, fn)

    logging.info('Loaded models for %d categories', len(models))

    stats = {cat: [0, 0, 0, 0] for cat in config.EXTRACTED_TAGS}

    for source, target, alignment in read_sentences(sys.stdin):
        for word, features in extract_instances(source, target, alignment):
            gold_inflection, lemma, tag = word
            category = tag[0]
            gold_tag = tag[1:]
            possible_inflections = rev_map.get((lemma, category), [])
            if (gold_tag, gold_inflection) not in possible_inflections:
                print(u'Expected: {} ({}) not found'.format(gold_inflection,
                    gold_tag).encode('utf8'))
                continue

            model = models[category]

            scored_inflections = model.score_all(possible_inflections, features)
            ranked_inflections = sorted(scored_inflections, reverse=True)
            predicted_score, predicted_tag, predicted_inflection = ranked_inflections[0]

            gold_rank = 1 + [tag for _, tag, _ in ranked_inflections].index(gold_tag)
            gold_score = next((score for score, tag, _ in ranked_inflections if tag == gold_tag))

            print(u'Expected: {} ({}) r={} score={:.3f} |'
                    ' Predicted: {} ({}) score={:.3f}'.format(gold_inflection,
                gold_tag, gold_rank, gold_score, predicted_inflection, predicted_tag,
                predicted_score).encode('utf8'))
            
            stats[category][0] += 1
            stats[category][1] += 1/float(gold_rank)
            stats[category][2] += (gold_inflection == predicted_inflection)
            stats[category][3] += gold_score

    for category, (n_instances, rrank_sum, n_correct, total_log_prob) in stats.items():
        if n_instances == 0: continue
        mrr = rrank_sum/n_instances
        accuracy = n_correct/float(n_instances)
        ppl = math.exp(-total_log_prob/n_instances)
        print('Category {}: MRR={:.3f} acc={:.3f} ppl={:.1f} ({:>4} instances)'.format(category, mrr, accuracy, ppl, n_instances))

if __name__ == '__main__':
    main()
