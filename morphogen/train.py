import sys
import argparse
import cPickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def sentences(stream):
    for line in stream:
        src, tgt, als = line.decode('utf8')[:-1].split(' ||| ')
        alignments = [(float(ij[0]), float(ij[1])) for ij in
            (point.split('=') for point in als.split())]
        yield src.split(), tgt.split(), alignments

def extract_instances(source, target, alignment):
    # analyze target with pymorphy
    # for each morphological feature of each word:
    # create feature vector (dictionary)
    # yield fvect, fval
    pass

def main():
    parser = argparse.ArgumentParser(description='Train morphology generation model')
    parser.add_argument('model', help='output file for trained model')
    args = parser.parse_args()

    training_features = []
    trainig_outputs = []
    for source, target, alignment in sentences(sys.stdin):
        for features, output in extract_instances(source, target, alignment):
            training_features.append(features)
            trainig_outputs.append(output)
    
    vectorizer = DictVectorizer()
    X, y = vectorizer.fit(training_features, trainig_outputs)
    model = LogisticRegression()
    model.fit(X, y)

    with open(args.model) as f:
        cPickle.dump(model, f)

if __name__ == '__main__':
    main()
