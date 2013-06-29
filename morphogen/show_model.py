import sys
import argparse
import cPickle
import heapq

def main():
    parser = argparse.ArgumentParser(description='Show trained model')
    parser.add_argument('model', help='trained model')
    parser.add_argument('--flat', action='store_true', help='do not group by morphfeat')
    args = parser.parse_args()

    with open(args.model) as f:
        model = cPickle.load(f)

    if args.flat:
        top = heapq.nlargest(100, ((output_feature, f, w)
            for output_feature in model.output_features
            for f, w in model.weights(output_feature)), key=lambda t: t[1])
        for output_feature, f, w in top:
            print(u'{}+{} = {}'.format(output_feature, f, w).encode('utf8'))
    else:
        for output_feature in model.output_features:
            sys.stdout.write(u'{}: '.format(output_feature).encode('utf8'))
            top = heapq.nlargest(10, model.weights(output_feature), key=lambda t: t[1])
            print(' '.join(u'{}={.2f}'.format(f, w) for f, w in top).encode('utf8'))

if __name__ == '__main__':
    main()
