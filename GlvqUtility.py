import numpy as np
from sklearn.utils import validation
from sklearn.utils.multiclass import unique_labels


def squared_euclidean(a, b=None):
    if b is None:
        d = np.sum(a ** 2, 1)[np.newaxis].T + np.sum(a ** 2, 1) - 2 * a.dot(
            a.T)
    else:
        d = np.sum(a ** 2, 1)[np.newaxis].T + np.sum(b ** 2, 1) - 2 * a.dot(
            b.T)
    return np.maximum(d, 0)


class GlvqUtility:

    def __init__(self, lvq_model):
        self.random_state = lvq_model.random_state
        self.initial_prototypes = lvq_model.initial_prototypes
        self.prototypes_per_class = lvq_model.prototypes_per_class
        self.display = lvq_model.display
        self.max_iter = lvq_model.max_iter
        self.gtol = lvq_model.gtol

    def validate_train_parms(self, train_set, train_lab):
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

    def get_prediction_accuracy(self, prototypes, train_x, train_y):

        x, y, _ = self.validate_train_parms(train_x, train_y)

        label_equals_prototype = y[np.newaxis].T == self.c_w_
        dist = squared_euclidean(x, prototypes)
        d_wrong = dist.copy()
        d_wrong[label_equals_prototype] = np.inf
        distwrong = d_wrong.min(1)

        d_correct = dist
        d_correct[np.invert(label_equals_prototype)] = np.inf
        distcorrect = d_correct.min(1)

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong

        return mu
