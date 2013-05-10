import sys
import argparse
import cPickle
import heapq

def main():
    parser = argparse.ArgumentParser(description='Show trained model')
    parser.add_argument('model', help='trained model')
    args = parser.parse_args()

    with open(args.model) as f:
        category, vectorizer, model = cPickle.load(f)

    for cls, weights in zip(model.classes_[:-1], model.coef_):
        sys.stdout.write(u'{}{}: '.format(category, cls).encode('utf8'))
        top = heapq.nlargest(10, vectorizer.inverse_transform(weights)[0].iteritems(), key=lambda t: t[1])
        print(' '.join(u'{}={}'.format(f, w) for f, w in top).encode('utf8'))

if __name__ == '__main__':
    main()

