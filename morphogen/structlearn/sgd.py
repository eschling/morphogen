import logging
import sys
import math
import numpy

def _add(w, u, a, b, alpha, adagrad=None, average=None, l1=0, t=None): #w += u*a.T b
    if a.nnz == 0 or b.nnz == 0: return
    for i, a_i in enumerate(a.indices):
        for j, b_j in enumerate(b.indices):
            if adagrad is None:
              w[a_i, b_j] += alpha * u * a.data[i] * b.data[j]
            else:
              grad = u * a.data[i] * b.data[j]
              adagrad[a_i, b_j] += grad**2
              average[a_i, b_j] += grad
              if adagrad[a_i, b_j] == 0: continue
              if not l1:
                w[a_i, b_j] += (alpha/math.sqrt(adagrad[a_i, b_j])) * grad
              else:
                z = abs(average[a_i, b_j])/t - l1
                #maximizing likelihood rather than minimizing log loss
                s = 1 if average[a_i, b_j]>0 else -1
                w[a_i, b_j] = ((alpha*t)/math.sqrt(adagrad[a_i, b_j])) * s * z if z>0 else 0

def _dot(w, a, b): # w.dot(a.T b)
    if a.nnz == 0 or b.nnz == 0: return 0
    return sum(w[a_i, b_j] * a.data[i] * b.data[j]
            for i, a_i in enumerate(a.indices)
            for j, b_j in enumerate(b.indices))

class StructuredClassifier:
    def __init__(self, n_iter=10, alpha_sgd=0.1):
        self.n_iter = n_iter
        self.alpha_sgd = alpha_sgd

    def fit(self, X, Y_all, Y_star, Y_lim=None, every_iter=None, Adagrad=False, l1_lambda=None):
        """
        X : CSR matrix (n_instances x n_features)
        Y_all : CSR matrix (n_outputs x n_labels)
        Y_star : 1d array (n_instances)
        Y_lim (optional) : array (n_instances x 2)
        """

        # Check dimensions
        n_instances, n_features = X.shape
        n_outputs, n_labels = Y_all.shape
        assert Y_star.shape == (n_instances, )
        if Y_lim is not None:
            assert Y_lim.shape == (n_instances, 2)

        self.weights = numpy.zeros((n_features, n_labels))
        self.y_weights = numpy.zeros((n_labels, n_labels))
        
        if Adagrad:
          adagrad = [numpy.zeros(self.weights.shape), numpy.zeros(self.y_weights.shape)]
          average = [numpy.zeros(self.weights.shape), numpy.zeros(self.y_weights.shape)]

        mod100 = max(1, n_instances/90)
        mod10 = max(1, n_instances/9)

        for it in xrange(self.n_iter):
            logging.info('Iteration %d/%d (rate=%s)', (it+1), self.n_iter, self.alpha_sgd)
            ll = 0
            for i in xrange(n_instances):
                if i % mod10 == 0: sys.stderr.write('|')
                elif i % mod100 == 0: sys.stderr.write('.')
                f, t = (Y_lim[i] if Y_lim is not None else (0, n_outputs)) # output limits
                Y_x = Y_all[f:t] # all compatible outputs
                x = X[i] # feature vector
                y_star = Y_star[i] # expected output
                # w.f(x, y) - log(Z(x))
                log_probs = self.predict_log_proba(x, Y_x)
                ll += log_probs[y_star]
                # exp(w.f(x, y)) / Z(x)
                probs = numpy.exp(log_probs)
                # - grad(loss) = + grad(LL) = x_star - sum_x(p(x) x)
                for y_i, y in enumerate(Y_x):
                    u = (int(y_i == y_star) - probs[y_i])
                    if u == 0: continue
                    if Adagrad:
                      _add(self.weights, u, x, y, self.alpha_sgd,
                           adagrad=adagrad[0], average=average[0], l1=l1_lambda, t=it+1)
                      _add(self.y_weights, u, y, y, self.alpha_sgd,
                           adagrad=adagrad[1], average=average[1], l1=l1_lambda, t=it+1)
                    else:
                      _add(self.weights, u, x, y, self.alpha_sgd)
                      _add(self.y_weights, u, y, y, self.alpha_sgd)
            sys.stderr.write('\n')
            logging.info('LL=%.3f ppl=%.3f', ll, math.exp(-ll/n_instances))
            if every_iter:
                every_iter(it, self)

    def predict_log_proba(self, x, Y_x):
        potentials = numpy.array([_dot(self.weights, x, y) + _dot(self.y_weights, y, y)
            for y in Y_x]) # w.f(x, y)
        return potentials - numpy.logaddexp.reduce(potentials)

    def predict(self, x, Y_x):
        return numpy.argmax(_dot(self.weights, x, y) + _dot(self.y_weights, y, y) for y in Y_x)
