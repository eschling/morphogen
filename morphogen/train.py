import sys
import argparse
import cPickle
import logging
from collections import defaultdict, namedtuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import config

AnnotatedToken = namedtuple('AnnotatedToken', 'token, pos, parent, dependency, cluster')

def read_sentences(stream):
    """Read annotated sentences in the format:
    EN ||| EN POS ||| EN dep ||| EN clus ||| RU ||| RU lemma ||| RU tag ||| alignment"""
    for line in stream:
        fields = line.decode('utf8')[:-1].split(' ||| ')
        src, src_pos, src_dep, src_clus, tgt, tgt_lem, tgt_tag, als = fields

        # Read target
        tgt = tgt.split()
        tgt_lem = tgt_lem.split()
        tgt_tag = tgt_tag.split()
        # check
        if not tgt_lem:
            logging.debug('Skip empty analysis for sentence: %s', ' '.join(tgt))
            continue
        if not len(tgt) == len(tgt_lem) == len(tgt_tag):
            logging.error('Bad annotation for "%s"', ' '.join(tgt))
            continue
        tgt_tokens = zip(tgt, tgt_lem, tgt_tag)

        # Read source
        src = src.split()
        src_pos = src_pos.split()
        src_parents, src_dtypes = zip(*[(int(parent), typ) for parent, typ in 
            (dep.split('-') for dep in src_dep.split())])
        src_clus = src_clus.split()
        src_tokens = [AnnotatedToken(*info) for info in
                zip(src, src_pos, src_parents, src_dtypes, src_clus)]

        if not len(src) == len(src_pos) == len(src_parents) == len(src_dtypes) == len(src_clus):
            logging.error('Bad tag/parse for "%s" (%d/%d/%d/%d/%d)', ' '.join(src),
                    len(src), len(src_pos), len(src_parents), len(src_dtypes), len(src_clus))
            continue

        # Read alignment (tgt - src) [ru - en]
        alignments = [(int(i), int(j)) for i, j in
            (point.split('-') for point in als.split())]

        yield src_tokens, tgt_tokens, alignments

def extract_instances(source, target, alignment):
    """Extract (category, features, tag) training instances for a sentence pair"""
    for i, (_, lemma, tag) in enumerate(target):
        if tag[0] not in config.EXTRACTED_TAGS: continue
        word_alignments = [j for (k, j) in alignment if k == i] # tgt == i - src
        if len(word_alignments) != 1: continue # Extract only one-to-one alignments
        (j,) = word_alignments # src
        features = dict((fname, fval) for ff in config.FEATURES
                for fname, fval in ff(source, lemma, j))
        yield tag[0], features, tag[1:]

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Train morphology generation model')
    parser.add_argument('model', help='output file for trained model')
    args = parser.parse_args()

    logging.info('Extracting features for training data')
    training_features = defaultdict(list)
    training_outputs = defaultdict(list)
    for source, target, alignment in read_sentences(sys.stdin):
        for category, features, output in extract_instances(source, target, alignment):
            training_features[category].append(features)
            training_outputs[category].append(output)

    models = {}
    for category in training_features:
        logging.info('Training model for category: %s', category)
        logging.info('Converting data into sparse matrix')
        vectorizer = DictVectorizer()
        X = vectorizer.fit_transform(training_features[category])
        y = training_outputs[category]
        logging.info('Training data size: %d instances x %d features', *X.shape)
        logging.info('Number of predicted tags: %d', len(set(y)))

        logging.info('Fitting model')
        model = LogisticRegression(C=0.01)
        model.fit(X, y)

        models[category] = (vectorizer, model)

    with open(args.model, 'w') as f:
        cPickle.dump(models, f, protocol=-1)

if __name__ == '__main__':
    main()
