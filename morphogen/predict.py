import sys
import argparse
import logging
import cPickle
import math
import config_files.config as config
from common import read_sentences
from models import load_models

def extract_instances(source, target, alignment, tags):
    """Extract (category, features, tag) training instances for a sentence pair"""
    for j, (token, lemma, tag) in enumerate(target):
        if tag[0] not in tags: continue
        word_alignments = [i for (i, k) in alignment if k == j] # src - tgt = j
        if len(word_alignments) != 1: continue # Extract only 1-n alignments
        (i,) = word_alignments # src
        features = dict((fname, fval) for ff in config.FEATURES
                for fname, fval in ff(source, lemma, i))
        yield (token, lemma, tag), features

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Predict using trained models')
    parser.add_argument('rev_map', help='reverse inflection map')
    parser.add_argument('models', nargs='+', help='trained models (category:file)')
    parser.add_argument('-a','--ambiguous', action='store_true',
            help='evaluate only lemmas with multiple inflections')
    args = parser.parse_args()

    logging.info('Loading reverse inflection map')
    with open(args.rev_map) as f:
        rev_map = cPickle.load(f)

    logging.info('Loading inflection prediction models')
    models = load_models(args.models)
    logging.info('Loaded models for %d categories', len(models))
    extract_tags = [cat for cat in models.keys()]

    stats = {cat: [0, 0, 0, 0, 0] for cat in extract_tags}

    for source, target, alignment in read_sentences(sys.stdin):
        for word, features in extract_instances(source, target, alignment, extract_tags):
            gold_inflection, lemma, tag = word
            category = tag[0]
            gold_tag = tag[1:]
            possible_inflections = rev_map.get((lemma, category), [])
            if (gold_tag, gold_inflection) not in possible_inflections:
                print(u'Expected: {} ({}) not found'.format(gold_inflection,
                    gold_tag).encode('utf8'))
                continue
            if args.ambiguous and len(possible_inflections) == 1: continue

            model = models[category]

            scored_inflections = model.score_all(possible_inflections, features)
            ranked_inflections = sorted(scored_inflections, reverse=True)
            predicted_score, predicted_tag, predicted_inflection = ranked_inflections[0]

            gold_rank = 1 + [tag for _, tag, _ in ranked_inflections].index(gold_tag)
            gold_score = next((score for score, tag, _ in ranked_inflections
                if tag == gold_tag))

            print(u'Expected: {} ({}) r={} score={:.3f} |'
                    ' Predicted: {} ({}) score={:.3f}'.format(gold_inflection,
                gold_tag, gold_rank, gold_score, predicted_inflection, predicted_tag,
                predicted_score).encode('utf8'))
            
            stats[category][0] += 1
            stats[category][1] += 1/float(gold_rank)
            stats[category][2] += (gold_inflection == predicted_inflection)
            stats[category][3] += gold_score
            stats[category][4] += len(ranked_inflections)

    for category, (n_instances, rrank_sum, n_correct, total_log_prob, n_inflections) in stats.items():
        if n_instances == 0: continue
        mrr = rrank_sum/n_instances
        accuracy = n_correct/float(n_instances)
        ppl = math.exp(-total_log_prob/n_instances)
        avg_inflections = n_inflections/float(n_instances)
        print('Category {}: MRR={:.3f} acc={:.1%} ppl={:.2f} ll={} ({} instances; avg #infl={:.2f})'.format(
            category, mrr, accuracy, ppl, total_log_prob, n_instances, avg_inflections))

if __name__ == '__main__':
    main()
