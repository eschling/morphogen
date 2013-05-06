import gzip, cPickle
import numpy
from collections import defaultdict
import tagset
from crf_train import get_attributes

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
        for attr in get_attributes(self.category, tag):
            for fname, fval in features.iteritems():
                score += fval * self.weights.get(attr+'_'+fname, 0)
        return score

    def score_all(self, inflections, features):
        scored = [(self.score(tag, features), tag, inflection)
                for tag, inflection in inflections]
        z = numpy.logaddexp.reduce([score for score, _, _ in scored])
        return [(score - z, tag, inflection) for score, tag, inflection in scored]

def load_models(model_files):
    models = {}
    for fn in model_files:
        if fn.endswith('.pickle'):
            with open(fn) as f:
                category, v, m = cPickle.load(f)
                models[category] = (VectorModel(category, v, m) if isinstance(m, dict)
                        else SimpleModel(v, m))
        elif fn.endswith('.gz'):
            category = fn[fn.find('.gz')-1]
            models[category] = CRFModel(category, fn)
