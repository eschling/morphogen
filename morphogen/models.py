import gzip, cPickle
import logging
from collections import defaultdict
import numpy
from sklearn.feature_extraction import DictVectorizer
import structlearn
import tagset # FIXME use config
import config

class SimpleModel:
    def __init__(self, vectorizer, clf):
        self.vectorizer = vectorizer
        self.clf = clf

    def score_all(self, inflections, features):
        fvector = self.vectorizer.transform(features)
        predictions = dict(zip(self.clf.classes_, self.clf.predict_log_proba(fvector)[0]))
        scored = [(predictions.get(tag, float('-inf')), tag, inflection)
                for tag, inflection in inflections]
        z = numpy.logaddexp.reduce([score for score, _, _ in scored])
        return [(score - z, tag, inflection) for score, tag, inflection in scored]

class VectorModel:
    def __init__(self, category, vectorizer, clfs):
        self.category = tagset.categories[category]
        self.vectorizer = vectorizer
        self.clfs = clfs

    def score_all(self, inflections, features):
        fvector = self.vectorizer.transform([features])
        score_vectors = {}
        for i, clf in self.clfs.iteritems():
            if clf is None: # univalued attribute
                score_vectors[i] = defaultdict(int)
            else:
                score_vectors[i] = dict(zip(clf.classes_, clf.predict_log_proba(fvector)[0]))
        score = lambda tag: sum(score_vectors[i][v]
                for i, v in enumerate(tag.ljust(tagset.tag_length[self.category], '-')))
        scored = [(score(tag), tag, inflection) for tag, inflection in inflections]
        z = numpy.logaddexp.reduce([score for score, _, _ in scored])
        return [(score - z, tag, inflection) for score, tag, inflection in scored]

class CRFModel:
    def __init__(self, category, fn):
        self.category = category
        self.weights = {}
        with gzip.open(fn) as f:
            for line in f:
                fname, fval = line.decode('utf8').split()
                self.weights[fname] = float(fval)

    def score(self, tag, features):
        score = 0
        for attr in config.get_attributes(self.category, tag):
            for fname, fval in features.iteritems():
                score += fval * self.weights.get(attr+'_'+fname, 0)
        return score

    def score_all(self, inflections, features):
        scored = [(self.score(tag, features), tag, inflection)
                for tag, inflection in inflections]
        z = numpy.logaddexp.reduce([score for score, _, _ in scored])
        return [(score - z, tag, inflection) for score, tag, inflection in scored]

class StructuredModel:
    def __init__(self, category):
        self.category = category
        self.feature_dict = DictVectorizer()
        self.label_dict = DictVectorizer()

    def train(self, X, Y_all, Y_star, Y_lim, n_iter=10,
            alpha_sgd=0.1, every_iter=None):
        logging.info('Converting into matrices')
        X = self.feature_dict.fit_transform(X)
        logging.info('X: %d x %d', *X.shape)
        Y_all = self.label_dict.fit_transform(Y_all)
        logging.info('Y_all: %d x %d', *Y_all.shape)
        Y_star = numpy.array(Y_star)
        logging.info('Y_star: %d', *Y_star.shape)
        Y_lim = numpy.array(Y_lim)
        logging.info('Y_lim: %d x %d', *Y_lim.shape)

        self.model = structlearn.StructuredClassifier(n_iter=n_iter, 
                alpha_sgd=alpha_sgd)
        if every_iter: # call every_iter with StructuredModel and not StructuredClassifier
            every_iter2 = lambda it, model: every_iter(it, self)
        else:
            every_iter2 = every_iter
        self.model.fit(X, Y_all, Y_star, Y_lim, every_iter=every_iter2)

    def score_all(self, inflections, features):
        X = self.feature_dict.transform([features])
        Y_all = []
        for i, (tag, _) in enumerate(inflections):
            label = {attr: 1 for attr in config.get_attributes(self.category, tag)}
            Y_all.append(label)
        Y_all = self.label_dict.transform(Y_all)

        scores = self.model.predict_log_proba(X, Y_all)
        return [(score, tag, inflection) for score, (tag, inflection)
                in zip(scores, inflections)]


def load_models(model_files):
    models = {}
    for fn in model_files:
        if fn.endswith('.pickle'):
            with open(fn) as f:
                model = cPickle.load(f)
            if isinstance(model, tuple): # old pickles
                category, v, m = model
                models[category] = (VectorModel(category, v, m) if isinstance(m, dict)
                    else SimpleModel(v, m))
            else:
                models[model.category] = model
        elif len(fn.split(':')) == 2:
            category, fn = fn.split(':')
            models[category] = CRFModel(category, fn)
    return models
