import sys
import argparse
import cPickle
import logging
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import analyzer
import basic_features

def read_sentences(stream):
    """Read source ||| target ||| alignments from stream"""
    for line in stream:
        src, tgt, als = line.decode('utf8')[:-1].split(' ||| ')
        alignments = [(float(ij[0]), float(ij[1])) for ij in
            (point.split('-') for point in als.split())]
        yield src.split(), tgt.split(), alignments

def annotate_sentences(tagger, sentences):
    """Annotate target with morphological tags"""
    for source, target, alignments in sentences:
        try:
            analyzed_target = list(tagger.tag(target))
        except analyzer.AnalysisError as e:
            logging.error('Tagging failed for "%s": %s', ' '.join(target), e)
            continue
        yield source, analyzed_target, alignments

def read_annotated(stream):
    """Read directly annotated data from stream in the format produce by pre-tag.py:
    source ||| target ||| target lemmas ||| target tags ||| alignment"""
    for line in stream:
        src, tgt, tgt_lem, tgt_ana, als = line.decode('utf8')[:-1].split(' ||| ')
        alignments = [(int(ij[0]), int(ij[1])) for ij in
            (point.split('-') for point in als.split())]
        tgt = tgt.split()
        tgt_lem = tgt_lem.split()
        tgt_ana = tgt_ana.split()
        if not (len(tgt) == len(tgt_lem) == len(tgt_ana)):
            logging.error('** Bad annotation for "%s"', ' '.join(tgt))
            continue
        analyzed_target = [analyzer.Analysis(*t) for t in zip(tgt, tgt_lem, tgt_ana)]
        yield src.split(), analyzed_target, alignments

# List of features function to use for extraction
FEATURES = [basic_features.bow]
# List of POS categories to extract training data for
EXTRACTED_TAGS = 'NVARM'

def extract_instances(source, analyses, alignment):
    """Extract (category, feature, tag) training instances for a sentence pair"""
    for j, analysis in enumerate(analyses):
        if analysis.tag[0] not in EXTRACTED_TAGS: continue
        word_alignments = [i for (i, k) in alignment if k == j]
        if len(word_alignments) != 1: continue # Extract only one-to-one alignments
        (i,) = word_alignments
        features = dict((fname, fval) for ff in FEATURES
                for fname, fval in ff(source, analysis.lemma, i))
        yield analysis.tag[0], features, analysis.tag[1:]

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
    training_features = defaultdict(list)
    training_outputs = defaultdict(list)
    for source, target, alignment in data:
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

        logging.info('Fitting model')
        model = LogisticRegression(C=0.1)
        model.fit(X, y)

        models[category] = (vectorizer, model)

    with open(args.model, 'w') as f:
        cPickle.dump(models, f, protocol=-1)

if __name__ == '__main__':
    main()
