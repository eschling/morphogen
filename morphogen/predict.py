import sys
import argparse
import logging
import cPickle
from train import read_sentences
import config

def extract_instances(source, target, alignment):
    """Extract (category, features, tag) training instances for a sentence pair"""
    for i, (token, lemma, tag) in enumerate(target):
        if tag[0] not in config.EXTRACTED_TAGS: continue
        word_alignments = [j for (k, j) in alignment if k == i] # tgt == i - src
        if len(word_alignments) != 1: continue # Extract only one-to-one alignments
        (j,) = word_alignments # src
        features = dict((fname, fval) for ff in config.FEATURES
                for fname, fval in ff(source, lemma, j))
        yield (token, lemma, tag), features

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

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
            vectorizer, model = m[category]
            fvector = vectorizer.transform(features)
            predictions = dict(zip(model.classes_, model.predict_proba(fvector)[0]))
            scored_inflections = ((predictions.get(tag, 0), tag, inflection)
                    for tag, inflection in possible_inflections)
            ranked_inflections = sorted(scored_inflections, reverse=True)
            predicted_prob, predicted_tag, predicted_inflection = ranked_inflections[0]

            gold_rank = 1 + [tag for _, tag, _ in ranked_inflections].index(gold_tag)
            gold_prob = predictions.get(gold_tag, 0)
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
