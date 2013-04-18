import argparse
import cPickle
import heapq

def main():
    parser = argparse.ArgumentParser(description='Show trained model')
    parser.add_argument('model', help='trained model')
    args = parser.parse_args()

    with open(args.model) as f:
        m = cPickle.load(f)
        vectorizer = m['vectorizer']
        model = m['model']

    fnames = vectorizer.get_feature_names()

    for cls, weights in zip(model.classes_[:-1], model.coef_):
        print cls
        top = heapq.nlargest(10, zip(weights, fnames))
        print(' '.join(u'{1}={0}'.format(*wf) for wf in top).encode('utf8'))

if __name__ == '__main__':
    main()

