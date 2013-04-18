import sys
import argparse
import cPickle
import logging
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import analyzer
import basic_features

def read_sentences(stream):
    for line in stream:
        src, tgt, als = line.decode('utf8')[:-1].split(' ||| ')
        alignments = [(float(ij[0]), float(ij[1])) for ij in
            (point.split('-') for point in als.split())]
        yield src.split(), tgt.split(), alignments

def annotate_sentences(tagger, sentences):
    for source, target, alignments in sentences:
        try:
            analyzed_target = list(tagger.tag(target))
        except analyzer.AnalysisError as e:
            logging.error('Tagging failed for "%s": %s', ' '.join(target), e)
            continue
        yield source, analyzed_target, alignments

def read_annotated(stream):
    for line in stream:
        src, tgt, tgt_lem, tgt_ana, als = line.decode('utf8')[:-1].split(' ||| ')
        alignments = [(float(ij[0]), float(ij[1])) for ij in
            (point.split('-') for point in als.split())]
        tgt = tgt.split()
        tgt_lem = tgt_lem.split()
        tgt_ana = tgt_ana.split()
        if not (len(tgt) == len(tgt_lem) == len(tgt_ana)):
            logging.error('** Bad annotation for "%s"', ' '.join(tgt))
            continue
        analyzed_target = [analyzer.Analysis(*t) for t in zip(tgt, tgt_lem, tgt_ana)]
        yield src.split(), analyzed_target, alignments

FEATURES = [basic_features.words]

def extract_instances(source, analyses, alignment):
    for j, analysis in enumerate(analyses):
        if analysis.tag[0] in ('-', ',', 'SENT', 'X'): continue
        features = dict((fname, fval) for ff in FEATURES
                for fname, fval in ff(source, j, analysis, alignment))
        yield features, analysis.tag

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Train morphology generation model')
    parser.add_argument('model', help='output file for trained model')
    parser.add_argument('--analyze', help='run morphological tagger')
    args = parser.parse_args()

    if args.analyze:
        tagger = analyzer.Tagger()
        data = annotate_sentences(tagger, read_sentences(sys.stdin))
    else:
        data = read_annotated(sys.stdin)

    logging.info('Extracting features for training data')
    training_features = []
    training_outputs = []
    for source, analyzed_target, alignment in data:
        for features, output in extract_instances(source, analyzed_target, alignment):
            training_features.append(features)
            training_outputs.append(output)

    logging.info('Converting into sparse matrix')
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(training_features)
    logging.info('Training data size: %d instances x %d features', *X.shape)

    logging.info('Fitting model')
    model = LogisticRegression(C=10)
    model.fit(X, training_outputs)

    with open(args.model, 'w') as f:
        cPickle.dump({'vectorizer': vectorizer, 'model': model}, f)

if __name__ == '__main__':
    main()
