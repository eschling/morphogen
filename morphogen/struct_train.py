import sys
import os
import argparse, logging
import cPickle
import config_files.config as config
import functools
from common import read_sentences
from crf_train import extract_instances
from models import StructuredModel

def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser(description='Trained stuctured model')
    parser.add_argument('category', help='target word category')
    parser.add_argument('rev_map', help='reverse inflection map')
    parser.add_argument('model', help='output directory for models')
    parser.add_argument('-i', '--n_iter', type=int, help='number of SGD iterations')
    parser.add_argument('-r', '--rate', type=float, help='SGD udpate rate')
    parser.add_argument('-c', '--config', help='configuration module for supervised models (must be in config directory)')
    args = parser.parse_args()

    logging.info('{}\n'.format(sys.argv))

    category = args.category
    logging.info('Training inflection model for category {}'.format(category))

    if args.config:
      logging.info('Loading external configuration module {}'.format(args.config))
      sup_config = __import__('config_files.'+args.config,globals(),locals(),['get_attributes'])
      attr_function = lambda cat, attr: sup_config.get_attributes(cat, attr)
    else:
      attr_function = lambda cat, attr: config.get_attributes(cat, attr)

    logging.info('Loading reverse inflection map')
    with open(args.rev_map) as f:
        rev_map = cPickle.load(f)

    logging.info('Generating the training data')
    X = []
    Y_all = []
    Y_star = []
    Y_lim = []
    n = 0
    inflection_lims = {} # inflection set cache (ranges for y in Y_all)
    for source, target, alignment in read_sentences(sys.stdin):
        for word, features in extract_instances(category, source, target, alignment):
            ref_inflection, lemma, tag = word
            category = tag[0]
            ref_attributes = tag[1:]
            possible_inflections = rev_map.get((lemma, category), [])
            # Skip if |inflections| = 1 [p(infl | lemma) = 1]
            if len(possible_inflections) == 1: continue
            if (ref_attributes, ref_inflection) not in possible_inflections: continue
            X.append(features)
            # Y_all / Y_lim
            lims = inflection_lims.get((lemma, category), None)
            if lims is None: # new set of inflections
                for i, (attributes, _) in enumerate(possible_inflections):
                    label = {attr: 1 for attr in attr_function(category, attributes)}
                    Y_all.append(label) # attributes map
                lims = (n, n+len(possible_inflections))
                inflection_lims[lemma, category] = lims
                n += len(possible_inflections)
            Y_lim.append(lims)
            # Y_star
            for i, (attributes, _) in enumerate(possible_inflections):
                if attributes == ref_attributes:
                    Y_star.append(i)

    # free some memory
    del rev_map

    if not os.path.exists(args.model):
        os.mkdir(args.model)
    def save_model(it, model):
        with open(os.path.join(args.model, 'model.{}.pickle'.format(it+1)), 'w') as f:
            cPickle.dump(model, f, protocol=-1)
    
    if args.config:
      model = StructuredModel(args.category, functools.partial(sup_config.get_attributes))
    else:
      model = StructuredModel(args.category, config.get_attributes)
    model.train(X, Y_all, Y_star, Y_lim, n_iter=args.n_iter,
            alpha_sgd=args.rate, every_iter=save_model)

if __name__ == '__main__':
    main()
