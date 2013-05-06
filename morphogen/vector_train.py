import sys
import argparse
import cPickle
import logging
from collections import defaultdict
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
import tagset
from common import read_sentences
from train import extract_instances

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Train morphology generation model')
    parser.add_argument('category', help='Russian word category to (R/V/A/N/M)')
    parser.add_argument('model', help='output file for trained model')
    parser.add_argument('--penalty', help='regularization penalty', type=float, default=0.001)
    args = parser.parse_args()

    assert len(args.category) == 1
    with open(args.model, 'w') as f:
        f.write('write test / training...')

    logging.info('Extracting features for training data')

    category = tagset.categories[args.category]

    training_features = defaultdict(list)
    training_outputs = defaultdict(list)
    for source, target, alignment in read_sentences(sys.stdin):
        for features, output in extract_instances(args.category, source, target, alignment):
            for i, v in enumerate(output.ljust(tagset.tag_length[category], '-')):
                training_features[i].append(features)
                training_outputs[i].append(v)

    logging.info('Training model for category: %s (%d attributes)', category, len(training_features))
    vectorizer = FeatureHasher()
    models = {}
    for i in training_features:
        X = vectorizer.transform(training_features[i])
        y = training_outputs[i]
        n_values = len(set(y))
        
        if n_values == 1:
            models[i] = None
            continue

        logging.info('Attribute %s (%d values)', tagset.attributes[category, i+1], n_values)
        logging.info('Training data size: %d instances x %d features', *X.shape)

        model = LogisticRegression(C=args.penalty)
        model.fit(X, y)
        models[i] = model

    with open(args.model, 'w') as f:
        cPickle.dump((args.category, vectorizer, models), f, protocol=-1)

if __name__ == '__main__':
    main()
