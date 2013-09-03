import logging
import sys
import math
import numpy
cimport numpy
cimport cython

@cython.boundscheck(False)
cdef _add(numpy.ndarray[numpy.float64_t, ndim=2] w, float u,
        numpy.ndarray[numpy.int32_t, ndim=1] a_indices,
        numpy.ndarray[numpy.float64_t, ndim=1] a_data,
        numpy.ndarray[numpy.int32_t, ndim=1] b_indices,
        numpy.ndarray[numpy.float64_t, ndim=1] b_data): # w += u * a.T b
    cdef unsigned i, a_i, j, b_j
    for i in range(a_indices.shape[0]):
        a_i = a_indices[i]
        for j in range(b_indices.shape[0]):
            b_j = b_indices[j]
            w[a_i, b_j] = w[a_i, b_j] + u * a_data[i] * b_data[j]

@cython.boundscheck(False)
cdef _add_grad(numpy.ndarray[numpy.float64_t, ndim=2] w, 
        float g, float alpha, float l1_lambda, int t,
        numpy.ndarray[numpy.float64_t, ndim=2] adagrad,
        numpy.ndarray[numpy.float64_t, ndim=2] u,
        numpy.ndarray[numpy.int32_t, ndim=1] a_indices,
        numpy.ndarray[numpy.float64_t, ndim=1] a_data,
        numpy.ndarray[numpy.int32_t, ndim=1] b_indices,
        numpy.ndarray[numpy.float64_t, ndim=1] b_data):
    cdef unsigned i, a_i, j, b_j
    for i in range(a_indices.shape[0]):
        a_i = a_indices[i]
        for j in range(b_indices.shape[0]):
            b_j = b_indices[j]
            grad = g * a_data[i] * b_data[j]
            adagrad[a_i, b_j] = adagrad[a_i, b_j] + grad**2
            u[a_i, b_j] = u[a_i, b_j] + grad
            if adagrad[a_i, b_j] == 0: continue
            if not l1_lambda: 
              w[a_i, b_j] = w[a_i, b_j] + (alpha/math.sqrt(adagrad[a_i, b_j])) * grad
            else:
              z = abs(u[a_i, b_j])/t - l1_lambda
              s = 1 if u[a_i, b_j]>0 else -1
              w[a_i, b_j] = ((alpha*t)/math.sqrt(adagrad[a_i, b_j])) * s * z if z>0 else 0

@cython.boundscheck(False)
cdef _dot(numpy.ndarray[numpy.float64_t, ndim=2] w,
        numpy.ndarray[numpy.int32_t, ndim=1] a_indices,
        numpy.ndarray[numpy.float64_t, ndim=1] a_data,
        numpy.ndarray[numpy.int32_t, ndim=1] b_indices,
        numpy.ndarray[numpy.float64_t, ndim=1] b_data): # w.dot(a.T b)
    cdef unsigned i, a_i, j, b_j
    cdef numpy.float64_t result = 0
    for i in range(a_indices.shape[0]):
        a_i = a_indices[i]
        for j in range(b_indices.shape[0]):
            b_j = b_indices[j]
            result += w[a_i, b_j] * a_data[i] * b_data[j]
    return result

class StructuredClassifier:
    def __init__(self, n_iter=10, alpha_sgd=0.1):
        self.n_iter = n_iter
        self.alpha_sgd = alpha_sgd

    def fit(self, X, Y_all, Y_star, Y_lim=None, every_iter=None, Adagrad=False, l1_lambda=0.0):
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

        self.weights = numpy.zeros((n_features, n_labels), dtype=numpy.float)
        self.y_weights = numpy.zeros((n_labels, n_labels), dtype=numpy.float)

        if Adagrad:
          adagrad = [numpy.zeros(self.weights.shape), numpy.zeros(self.y_weights.shape)]
          u = [numpy.zeros(self.weights.shape), numpy.zeros(self.y_weights.shape)]

        mod100 = max(1, n_instances/90)
        mod10 = max(1, n_instances/9)

        for it in xrange(self.n_iter):
            logging.info('Iteration %d/%d (rate=%s)', (it+1), self.n_iter, self.alpha_sgd)
            ll = 0
            for i in xrange(n_instances):
                if i % mod10 == 0: sys.stderr.write('| ll:{}'.format(ll))
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
                    grad = (int(y_i == y_star) - probs[y_i])
                    if grad == 0: continue
                    if Adagrad:
                      _add_grad(self.weights, grad, self.alpha_sgd, l1_lambda, it+1, adagrad[0], u[0], x.indices, x.data, y.indices, y.data)
                      _add_grad(self.y_weights, grad, self.alpha_sgd, l1_lambda, it+1, adagrad[1], u[1], y.indices, y.data, y.indices, y.data)
                    else:
                      _add(self.weights, grad*self.alpha_sgd, x.indices, x.data, y.indices, y.data)
                      _add(self.y_weights, grad*self.alpha_sgd, y.indices, y.data, y.indices, y.data)

            sys.stderr.write('\n')
            logging.info('LL=%.3f ppl=%.3f', ll, math.exp(-ll/n_instances))
            if every_iter:
                every_iter(it, self)

    def predict_log_proba(self, x, Y_x):
        cdef numpy.ndarray[numpy.float64_t, ndim=1] potentials
        cdef unsigned i
        cdef numpy.float64_t z, v
        potentials = numpy.zeros(Y_x.shape[0], dtype=numpy.float)
        for i in range(Y_x.shape[0]):
            y = Y_x[i]
            # x.T * W[xy] * y + y.T * W[yy] * y
            v = (_dot(self.weights, x.indices, x.data, y.indices, y.data)
                + _dot(self.y_weights, y.indices, y.data, y.indices, y.data))
            z = (v if i == 0 else numpy.logaddexp(z, v)) # partition function
            potentials[i] = v
        return potentials - z

    def predict(self, x, Y_x):
        return numpy.argmax(self.predict_log_proba(x, Y_x))
