import sys
import argparse, logging
import cPickle, gzip
import config
from train import read_sentences
from crf_train import get_attributes
from predict import extract_instances

class CRFModel:
    def __init__(self, fn):
        self.weights = {}
        with gzip.open(fn) as f:
            for line in f:
                fname, fval = line.decode('utf8').split()
                self.weights[fname] = float(fval)

    def score(self, category, tag, features):
        score = 0
        for attr in get_attributes(category, tag):
            for fname, fval in features.iteritems():
                score += fval * self.weights.get(attr+'_'+fname, 0)
        return score

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
        models[category] = CRFModel(fn)

    logging.info('Loaded models for %d categories', len(models))

    stats = {cat: [0, 0] for cat in config.EXTRACTED_TAGS}

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
            scored_inflections = ((model.score(category, tag, features), tag, inflection)
                    for tag, inflection in possible_inflections)
            ranked_inflections = sorted(scored_inflections, reverse=True)
            predicted_prob, predicted_tag, predicted_inflection = ranked_inflections[0]

            gold_rank = 1 + [tag for _, tag, _ in ranked_inflections].index(gold_tag)
            gold_prob = models[category].score(category, gold_tag, features) # TODO normalize
            print(u'Expected: {} ({}) r={} p={:.3f} |'
                    ' Predicted: {} ({}) p={:.3f}'.format(gold_inflection,
                gold_tag, gold_rank, gold_prob, predicted_inflection, predicted_tag,
                predicted_prob).encode('utf8'))
            
            stats[category][0] += 1
            stats[category][1] += 1/float(gold_rank)

    for category, (n_instances, rrank_sum) in stats.items():
        if n_instances == 0: continue
        mrr = rrank_sum/n_instances
        print('Category {}: {} instances -> MRR={:.3f}'.format(category, n_instances, mrr))

if __name__ == '__main__':
    main()
