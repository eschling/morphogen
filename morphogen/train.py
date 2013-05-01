import sys
import argparse
import cPickle
import logging
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import config
from common import read_sentences

def extract_instances(category, source, target, alignment):
    """Extract (features, tag) training instances for a sentence pair"""
    for i, (_, lemma, tag) in enumerate(target):
        if tag[0] != category: continue
        word_alignments = [j for (k, j) in alignment if k == i] # tgt == i - src
        if len(word_alignments) != 1: continue # Extract only one-to-one alignments
        (j,) = word_alignments # src
        features = dict((fname, fval) for ff in config.FEATURES
                for fname, fval in ff(source, lemma, j))
        yield features, tag[1:]

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

    training_features = []
    training_outputs = []
    for source, target, alignment in read_sentences(sys.stdin):
        for features, output in extract_instances(args.category, source, target, alignment):
            training_features.append(features)
            training_outputs.append(output)

    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(training_features)
    y = training_outputs

    logging.info('Training data size: %d instances x %d features', *X.shape)
    logging.info('Training model for category: %s (%d tags)', args.category, len(set(y)))

    model = LogisticRegression(C=args.penalty)
    model.fit(X, y)

    with open(args.model, 'w') as f:
        cPickle.dump((args.category, vectorizer, model), f, protocol=-1)

if __name__ == '__main__':
    main()
