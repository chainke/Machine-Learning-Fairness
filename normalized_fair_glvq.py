

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause

from __future__ import division

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from fair_glvq import mean_difference
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import validation
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
import operator


def _squared_euclidean(a, b=None):
    if b is None:
        d = np.sum(a ** 2, 1)[np.newaxis].T + np.sum(a ** 2, 1) - 2 * a.dot(
            a.T)
    else:
        d = np.sum(a ** 2, 1)[np.newaxis].T + np.sum(b ** 2, 1) - 2 * a.dot(
            b.T)
    return np.maximum(d, 0)


def normalized_mean_difference(protected_labels, nr_protected, dist):
    m = len(protected_labels)
    sgd_unprotected, sgd_protected = fairness_phi(protected_labels,dist)
    min_index, min_value = minimum_norm(sgd_protected+sgd_unprotected,m , nr_protected)
    norm_mean_difference = (sgd_unprotected/(m-nr_protected)-sgd_protected/nr_protected)/min_value
    return norm_mean_difference


def minimum_norm(sum_phi, m, nr_protected):
    values = [(sum_phi / m) / (1 - nr_protected / m), (1 - sum_phi / m) / (nr_protected / m)]

    if values[0] == values[1]:
        min_index = 2
        min_value = values[0]
    else:
        min_index, min_value = min(enumerate(values), key=operator.itemgetter(1))
    return min_index, min_value


def fairness_phi(protected_labels, dist):
    phi_protected_group = 0
    phi_unprotected_group = 0
    for i in range(0, len(protected_labels)):
        d0 = dist[i][0]
        d1 = dist[i][1]
        mu = sgd((d0 - d1) / (d0 + d1))
        phi_protected_group += protected_labels[i] * mu
        phi_unprotected_group += (1 - protected_labels[i]) * mu
    return phi_unprotected_group, phi_protected_group


def sgd(x):
    return 1 / (1 + np.exp(-x))


def dsgd(x):
    return np.exp(-x) / ((np.exp(-x) + 1) ** 2)

#=============================================================================================================
class GlvqModel(BaseEstimator, ClassifierMixin):
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

    def __init__(self, prototypes_per_class=1, initial_prototypes=None,
                 max_iter=2500, gtol=1e-5,
                 display=False, random_state=None):
        self.random_state = random_state
        self.initial_prototypes = initial_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.display = display
        self.max_iter = max_iter
        self.gtol = gtol

    def _optgrad(self, variables, training_data, label_equals_prototype,
                 random_state):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = variables.reshape(nb_prototypes, n_dim)

        dist = _squared_euclidean(training_data, prototypes)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)
        pidxwrong = d_wrong.argmin(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)
        pidxcorrect = d_correct.argmin(1)

        distcorrectpluswrong = distcorrect + distwrong

        g = np.zeros(prototypes.shape)

        distcorrectpluswrong = 4 / distcorrectpluswrong ** 2

        for i in range(nb_prototypes):
            idxc = i == pidxcorrect
            idxw = i == pidxwrong

            dcd = distcorrect[idxw] * distcorrectpluswrong[idxw]
            dwd = distwrong[idxc] * distcorrectpluswrong[idxc]
            g[i] = dcd.dot(training_data[idxw]) - dwd.dot(
                training_data[idxc]) + (dwd.sum(0) -
                                        dcd.sum(0)) * prototypes[i]
        g[:nb_prototypes] = 1 / n_data * g[:nb_prototypes]
        g = g * (1 + 0.0001 * random_state.rand(*g.shape) - 0.5)
        return g.ravel()

    def _optfun(self, variables, training_data, label_equals_prototype):
        n_data, n_dim = training_data.shape
        nb_prototypes = self.c_w_.size
        prototypes = variables.reshape(nb_prototypes, n_dim)

        dist = _squared_euclidean(training_data, prototypes)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong

        return mu.sum(0)

    def _validate_train_parms(self, train_set, train_lab):
        random_state = validation.check_random_state(self.random_state)
        if not isinstance(self.display, bool):
            raise ValueError("display must be a boolean")
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be an positive integer")
        if not isinstance(self.gtol, float) or self.gtol <= 0:
            raise ValueError("gtol must be a positive float")
        train_set, train_lab = validation.check_X_y(train_set, train_lab)

        self.classes_ = unique_labels(train_lab)
        nb_classes = len(self.classes_)
        nb_samples, nb_features = train_set.shape  # nb_samples unused

        # set prototypes per class
        if isinstance(self.prototypes_per_class, int):
            if self.prototypes_per_class < 0 or not isinstance(
                    self.prototypes_per_class, int):
                raise ValueError("prototypes_per_class must be a positive int")
            nb_ppc = np.ones([nb_classes],
                             dtype='int') * self.prototypes_per_class
        else:
            nb_ppc = validation.column_or_1d(
                validation.check_array(self.prototypes_per_class,
                                       ensure_2d=False, dtype='int'))
            if nb_ppc.min() <= 0:
                raise ValueError(
                    "values in prototypes_per_class must be positive")
            if nb_ppc.size != nb_classes:
                raise ValueError(
                    "length of prototypes per class"
                    " does not fit the number of classes"
                    "classes=%d"
                    "length=%d" % (nb_classes, nb_ppc.size))
        # initialize prototypes
        if self.initial_prototypes is None:
            self.w_ = np.empty([np.sum(nb_ppc), nb_features], dtype=np.double)
            self.c_w_ = np.empty([nb_ppc.sum()], dtype=self.classes_.dtype)
            pos = 0
            for actClass in range(nb_classes):
                nb_prot = nb_ppc[actClass]
                mean = np.mean(
                    train_set[train_lab == self.classes_[actClass], :], 0)
                self.w_[pos:pos + nb_prot] = mean + (
                    random_state.rand(nb_prot, nb_features) * 2 - 1)
                self.c_w_[pos:pos + nb_prot] = self.classes_[actClass]
                pos += nb_prot
        else:
            x = validation.check_array(self.initial_prototypes)
            self.w_ = x[:, :-1]
            self.c_w_ = x[:, -1]
            if self.w_.shape != (np.sum(nb_ppc), nb_features):
                raise ValueError("the initial prototypes have wrong shape\n"
                                 "found=(%d,%d)\n"
                                 "expected=(%d,%d)" % (
                                     self.w_.shape[0], self.w_.shape[1],
                                     nb_ppc.sum(), nb_features))
            if set(self.c_w_) != set(self.classes_):
                raise ValueError(
                    "prototype labels and test data classes do not match\n"
                    "classes={}\n"
                    "prototype labels={}\n".format(self.classes_, self.c_w_))
        return train_set, train_lab, random_state

    def _optimize(self, x, y, random_state):
        label_equals_prototype = y[np.newaxis].T == self.c_w_
        res = minimize(
            fun=lambda vs: self._optfun(
                variables=vs, training_data=x,
                label_equals_prototype=label_equals_prototype),
            jac=lambda vs: self._optgrad(
                variables=vs, training_data=x,
                label_equals_prototype=label_equals_prototype,
                random_state=random_state),
            method='l-bfgs-b', x0=self.w_,
            options={'disp': self.display, 'gtol': self.gtol,
                     'maxiter': self.max_iter})
        self.w_ = res.x.reshape(self.w_.shape)
        self.n_iter_ = res.nit

    def fit(self, x, y):
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
        self._optimize(x, y, random_state)
        return self

    def _compute_distance(self, x, w=None):
        if w is None:
            w = self.w_
        return cdist(x, w, 'euclidean')

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

# =============================================================================================================
# protected lables 1 = in protected class

    def gradient_norm_mean_difference(self, protected_labels, dist, data):
        m = len(protected_labels)
        nr_protected = sum(protected_labels)
        g = []

        sgd_unprotected, sgd_protected = fairness_phi(protected_labels, dist)
        sum_phi = sgd_protected+sgd_unprotected

        min_index, min_value = minimum_norm(sum_phi, m, nr_protected)
        dwi_min = self.gradient_minimum(dist, data, nr_protected, min_index)
        mean_diff = mean_difference(protected_labels, dist, data)
        dwi_mean_diff = self.gradient_mean_difference(protected_labels, dist, data)
        for i in range(0,1):
            dwi = (dwi_mean_diff[i]*min_value-mean_diff*dwi_min[i])/(min_value ** 2)
            g.append(dwi)
        return g

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
        summe = np.zeros(len(data[0]))
        vz = wi * 2 - 1
        for i in range(0, len(data)):
            # TODO: Hardcoded "negative" class, should be adjustable later.
            d0 = dist[i][0]
            d1 = dist[i][1]
            drel = dist[i][1 - wi]
            mu = (d0 - d1) / (d0 + d1)
            summe += dsgd(mu) * (4 * drel / (d0 + d1) ** 2) * (data[i] - self.w_[wi])
        return vz * summe / nr_data

    # We normalize through min(A,B) and therefore need the partial derivatives of A, B, and 1/2(A+B).
    # The following methods compute the derivatives.

    def dwi_minimum_a(self, dist, data, nr_protected, wi):
        m = len(data)
        dwi_min_a = self.dwi_mean_difference(1, dist, data, wi)/(m-nr_protected)
        return dwi_min_a

    def dwi_minimum_b(self, dist, data, nr_protected, wi):
        dwi_min_b = -self.dwi_mean_difference(1, dist, data, wi)/(nr_protected)
        return dwi_min_b

    def gradient_minimum(self, dist, data, nr_protected, case):
        dwi_min = []
        for i in range(0,1):
            if case == 0:
                dwi_min.append(self.dwi_minimum_a(dist, data, nr_protected, i))
            if case == 1:
                dwi_min.append(self.dwi_minimum_b(dist, data, nr_protected, i))
            else:
                dwi_min.append(0.5*(self.dwi_minimum_a(dist, data, nr_protected, i)
                            + self.dwi_minimum_b(dist, data, nr_protected, i)))
        return dwi_min