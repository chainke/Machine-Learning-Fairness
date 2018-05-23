# -*- coding: utf-8 -*-

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause

from __future__ import division

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from itertools import product
from sklearn.utils import validation
from sklearn.utils.validation import check_is_fitted

from sklearn_lvq.lvq import _LvqBaseModel


def _squared_euclidean(a, b=None):
    if b is None:
        d = np.sum(a ** 2, 1)[np.newaxis].T + np.sum(a ** 2, 1) - 2 * a.dot(
            a.T)
    else:
        d = np.sum(a ** 2, 1)[np.newaxis].T + np.sum(b ** 2, 1) - 2 * a.dot(
            b.T)
    return np.maximum(d, 0)


def sgd(x, beta):
    return 1 / (1 + np.exp(- beta*x))


def dsgd(x, beta):
    return beta*np.exp(-beta*x) / ((np.exp(-beta*x) + 1) ** 2)


def mean_difference(protected_labels, nr_protected_group, dist, beta):
    nr_non_protected_group = len(protected_labels) - nr_protected_group
    sgd_positive_class = 0
    sgd_negative_class = 0
    for i in range(0, len(protected_labels)):
        d0 = dist[i][0]
        d1 = dist[i][1]
        mu = sgd((d0 - d1) / (d0 + d1), beta)
        sgd_positive_class += protected_labels[i] * mu
        sgd_negative_class += (1 - protected_labels[i]) * mu

    fairnessdiff = sgd_positive_class / nr_protected_group - sgd_negative_class / nr_non_protected_group
    return fairnessdiff


class MeanDiffGlvqModel(_LvqBaseModel):
    """Generalized Learning Vector Quantization

    Parameters
    ----------

    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different
        numbers per class.

    initial_prototypes : array-like, shape =  [n_prototypes, n_features + 1],
     optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype.

    alpha : percentage of fairness relevance.
        alpha = 0 means normal glvq.

    max_iter : int, optional (default=2500)
        The maximum number of iterations.

    gtol : float, optional (default=1e-5)
        Gradient norm must be less than gtol before successful termination
        of bfgs.

    display : boolean, optional (default=False)
        Print information about the bfgs steps.

    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------

    w_ : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features

    c_w_ : array-like, shape = [n_prototypes]
        Prototype classes

    classes_ : array-like, shape = [n_classes]
        Array containing labels.

    See also
    --------
    GrlvqModel, GmlvqModel, LgmlvqModel
    """

    # def __init__(self, alpha=0, prototypes_per_class=1, initial_prototypes=None,
    #              max_iter=2500, gtol=1e-5, beta=2,
    #              display=False, random_state=None):
    #     self.random_state = random_state
    #     self.initial_prototypes = initial_prototypes
    #     self.prototypes_per_class = prototypes_per_class
    #     self.display = display
    #     self.max_iter = max_iter
    #     self.gtol = gtol
    #     self.alpha = alpha
    #     self.beta = beta

    def __init__(self, alpha=0, prototypes_per_class=1, initial_prototypes=None,
                 max_iter=2500, gtol=1e-5, beta=2, C=None,
                 display=False, random_state=None):
        super(MeanDiffGlvqModel, self).__init__(prototypes_per_class=prototypes_per_class,
                                                initial_prototypes=initial_prototypes,
                                                max_iter=max_iter, gtol=gtol, display=display,
                                                random_state=random_state)
        self.beta = beta
        self.c = C
        self.alpha = alpha

    def phi(self, x):
        """
        Parameters
        ----------

        x : input value

        """
        return 1 / (1 + np.exp(-self.beta * x))

    def phi_prime(self, x):
        """
        Parameters
        ----------

        x : input value

        """
        return self.beta * np.exp(self.beta * x) / (
                1 + np.exp(self.beta * x)) ** 2

    def _optgrad(self, variables, training_data, label_equals_prototype,
                 random_state, protected_labels, nr_protected_group):
        # --------------------------------------------------------------
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = variables.reshape(nb_prototypes, n_dim)

        dist = _squared_euclidean(training_data, prototypes)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)
        pidxwrong = d_wrong.argmin(1)

        d_correct = dist.copy()
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)
        pidxcorrect = d_correct.argmin(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong

        fair_diff = mean_difference(protected_labels, nr_protected_group, dist, self.beta)
        fair_dw = self.gradient_mean_difference(protected_labels, dist, training_data)
        distcorrectpluswrong = 4 / distcorrectpluswrong ** 2

        mu = np.vectorize(self.phi_prime)(mu)
        g = np.zeros(prototypes.shape)

        for i in range(nb_prototypes):
            idxc = i == pidxcorrect
            idxw = i == pidxwrong

            dcd = mu[idxw] * distcorrect[idxw] * distcorrectpluswrong[idxw]
            dwd = mu[idxc] * distwrong[idxc] * distcorrectpluswrong[idxc]
            g[i] = dcd.dot(training_data[idxw]) - dwd.dot(
                training_data[idxc]) + (dwd.sum(0) -
                                        dcd.sum(0)) * prototypes[i]
            g[i] += self.alpha * 2 * fair_diff * fair_dw[i]

        g[:nb_prototypes] = 1 / n_data * g[:nb_prototypes]
        g = g * (1 + 0.0001 * random_state.rand(*g.shape) - 0.5)
        return g.ravel()

    def _optfun(self, variables, training_data, label_equals_prototype, protected_labels, nr_protected_group):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = variables.reshape(nb_prototypes, n_dim)

        dist = _squared_euclidean(training_data, prototypes)
        # print(dist)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)

        d_correct = dist.copy()
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong

        [self._map_to_int(x) for x in self.c_w_[label_equals_prototype.argmax(1)]]
        mu *= self.c_[label_equals_prototype.argmax(1), d_wrong.argmin(1)]  # y_real, y_pred

        mu_sum = np.vectorize(self.phi)(mu).sum(0)

        if self.alpha == 0:
            return mu_sum

        error_normal = mu_sum / len(training_data)
        error_fairness = self.alpha * (mean_difference(protected_labels, nr_protected_group, dist, self.beta) ** 2)
        return error_normal + error_fairness

    def _validate_train_parms(self, train_set, train_lab):
        if not isinstance(self.beta, int):
            raise ValueError("beta must a an integer")

        ret = super(MeanDiffGlvqModel, self)._validate_train_parms(train_set, train_lab)

        self.c_ = np.ones((self.c_w_.size, self.c_w_.size))
        if self.c is not None:
            self.c = validation.check_array(self.c)
            if self.c.shape != (2, 3):
                raise ValueError("C must be shape (2,3)")
            for k1, k2, v in self.c:
                self.c_[tuple(zip(*product(self._map_to_int(k1), self._map_to_int(k2))))] = float(v)
        return ret

    def _map_to_int(self, item):
        return np.where(self.c_w_ == item)[0]

    def _optimize(self, x, y, protected_labels, random_state):
        label_equals_prototype = y[np.newaxis].T == self.c_w_
        nr_protected_group = sum(protected_labels,0)
        res = minimize(
            fun=lambda vs: self._optfun(
                variables=vs, training_data=x,
                label_equals_prototype=label_equals_prototype, protected_labels=protected_labels,
                nr_protected_group=nr_protected_group),
            jac=lambda vs: self._optgrad(
                variables=vs, training_data=x,
                label_equals_prototype=label_equals_prototype,
                random_state=random_state, protected_labels=protected_labels, nr_protected_group=nr_protected_group),
            method='l-bfgs-b', x0=self.w_,
            options={'disp': self.display, 'gtol': self.gtol,
                     'maxiter': self.max_iter})
        self.w_ = res.x.reshape(self.w_.shape)
        self.n_iter_ = res.nit

    def _compute_distance(self, x, w=None):
        if w is None:
            w = self.w_
        return cdist(x, w, 'euclidean')

    def split_x(self, x, dim_protected):

        protected = []
        new_x = []

        for i in range(0, len(x)):
            protected.append(x[i][dim_protected])
            new_x.append(
                x[i][:dim_protected] + x[i][dim_protected + 1:]
            )

        return new_x, protected

    def fit_fair(self, x, y, protected_labels):
        """Fit the GLVQ model to the given training data and parameters using
        l-bfgs-b.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
          Training vector, where n_samples in the number of samples and
          n_features is the number of features.
        y : array, shape = [n_samples]
          Target values (integers in classification, real numbers in
          regression)

        Returns
        --------
        self
        """
        x, y, random_state = self._validate_train_parms(x, y)
        if len(np.unique(y)) == 1:
            raise ValueError("fitting " + type(
                self).__name__ + " with only one class is not possible")

        self._optimize(x, y, protected_labels, random_state)
        return self

    def gradient_mean_difference(self, protected_labels, dist, data):
        nr_protected = sum(protected_labels)
        nr_unprotected = len(data) - nr_protected
        dist_protected = []
        data_protected = []
        dist_unprotected = []
        data_unprotected = []

        for i in range(0, len(data)):
            if protected_labels[i] == 0:
                dist_unprotected.append(dist[i])
                data_unprotected.append(data[i])
            else:
                dist_protected.append(dist[i])
                data_protected.append(data[i])
        dw0 = self.dwi_mean_difference(nr_protected, dist_protected, data_protected, 0) - self.dwi_mean_difference(
            nr_unprotected, dist_unprotected, data_unprotected, 0)
        dw1 = self.dwi_mean_difference(nr_protected, dist_protected, data_protected, 1) - self.dwi_mean_difference(
            nr_unprotected, dist_unprotected, data_unprotected, 1)
        return [dw0, dw1]

    def dwi_mean_difference(self, nr_data, dist, data, wi):
        sum = np.zeros(len(data[0]))
        vz = wi * 2 - 1
        for i in range(0, len(data)):
            # TODO: Hardcoded "negative" class, should be adjustable later.
            d0 = dist[i][0]
            d1 = dist[i][1]
            drel = dist[i][1 - wi]
            mu = (d0 - d1) / (d0 + d1)
            sum += dsgd(mu, self.beta) * (4 * drel / (d0 + d1) ** 2) * (data[i] - self.w_[wi])
        return vz * sum / nr_data

    def predict(self, x):
        """Predict class membership index for each input sample.

        This function does classification on an array of
        test vectors X.


        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]


        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self, ['w_', 'c_w_'])
        x = validation.check_array(x)
        if x.shape[1] != self.w_.shape[1]:
            raise ValueError("X has wrong number of features\n"
                             "found=%d\n"
                             "expected=%d" % (self.w_.shape[1], x.shape[1]))
        dist = self._compute_distance(x)
        return (self.c_w_[dist.argmin(1)])