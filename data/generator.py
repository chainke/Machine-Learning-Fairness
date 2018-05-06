import math
import numpy as np

import matplotlib.pyplot as plt
from sklearn_lvq.utils import _tango_color

# from glvq.plot_2d import to_tango_colors
# from glvq.plot_2d import plot2d

from sklearn.utils import shuffle
from sklearn_lvq.glvq import GlvqModel
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
from quad_fair_glvq import MeanDiffGlvqModel as FairGlvqModel
from normalized_fair_glvq import NormMeanDiffGlvqModel as NormFairGlvqModel


class DataGen:
    def __init__(self, verbose=False):
        """
            Constructor. More needed?

            Parameters
            ----------
            verbose: bool
                Flag to show status prints.

        """
        # number of data points     - Likely
        # proportion of subclasses  - Likely
        # number of features        - Maybe later?
        # number of subclasses      - necessary???
        self.verbose = verbose

        self.color_0 = 'skyblue'
        self.color_1 = 'scarletred'
        self.color_pos = 'chocolate'

    def generate_two_bubbles(self, number_data_points, proportion_0, proportion_0_urban, proportion_1_urban,
                             proportion_0_pay, proportion_1_pay, std_feature_1=0.2, std_feature_2=0.2):
        """
            Generates data set around two bubbles (normal distributions).

            Parameters
            ----------
            number_data_points: int
                Size of the data set.

            proportion_0: float in interval [0, 1]
                Proportion of the protected subclass.

            proportion_0_urban: float in interval [0, 1]
                Proportion of the protected subclass in urban neighbourhoods.

            proportion_1_urban: float in interval [0, 1]
                Proportion of the un-protected subclass in urban neighbourhoods.

            proportion_0_pay: float in interval [0, 1]
                Proportion of the protected subclass that repays their credit.

            proportion_1_pay: float in interval [0, 1]
                Proportion of the un-protected subclass that repays their credit.

            std_feature_1: float in interval [0, 1]
                Standard deviation of first feature in normal distributions to create bubbles.

            std_feature_2: float in interval [0, 1]
                Standard deviation of second feature in normal distributions to create bubbles.

            Returns
            -------
            r : float
                measure of discrimination
            X_full: np.array of float
                Features of the generated data set.
            C_full: np.array of bool
                Class membership of the protected class.
            Y_full: np.array of bool
                Outcomes.

        """

        std_array = np.array([std_feature_1, std_feature_2])

        X, C, Y = self.generate_two_bubbles_multi_dim(number_data_points, proportion_0, proportion_0_urban,
                                                      proportion_1_urban, proportion_0_pay, proportion_1_pay, std_array)

        return X, C, Y

    # NOT RUNNING YET!
    def generate_two_bubbles_multi_dim(self, number_data_points, proportion_0, proportion_0_urban, proportion_1_urban,
                                       proportion_0_pay, proportion_1_pay, std):
        """
            Generates data set around two bubbles (normal distributions).
            Important:  The second feature is still used for shift and to determine
                        the membership of the protected variable C.

            Parameters
            ----------
            number_data_points: int
                Size of the data set.

            proportion_0: float in interval [0, 1]
                Proportion of the protected subclass.

            proportion_0_urban: float in interval [0, 1]
                Proportion of the protected subclass in urban neighbourhoods.

            proportion_1_urban: float in interval [0, 1]
                Proportion of the un-protected subclass in urban neighbourhoods.

            proportion_0_pay: float in interval [0, 1]
                Proportion of the protected subclass that repays their credit.

            proportion_1_pay: float in interval [0, 1]
                Proportion of the un-protected subclass that repays their credit.

            std: np.array of float in interval [0, 1]
                Standard deviations for each feature in normal distributions to create bubbles.

            Returns
            -------
            r : float
                measure of discrimination
                measure of discrimination
            X_full: np.array of float
                Features of the generated data set.
            C_full: np.array of bool
                Class membership of the protected class.
            Y_full: np.array of bool
                Outcomes.

        """

        # proportion of non-white people who do not pay their money back
        proportion_0_dont_pay = 1 - proportion_0_pay
        # proportion of white people who do not pay their money back
        proportion_1_dont_pay = 1 - proportion_1_pay

        # proportion of non-white people in suburban neighbourhoods
        proportion_0_suburban = 1 - proportion_0_urban

        # number of people living in urban neighbourhoods
        m_urban = int(number_data_points * (1 - proportion_0) * proportion_1_urban +
                      number_data_points * proportion_0 * proportion_0_urban)
        # number of people living in suburban neighbourhoods
        m_suburb = number_data_points - m_urban
        if self.verbose:
            print("----")
            print("Total number of people in urban neighbourhoods: \t{}".format(m_urban))
            print("Total number of people in suburban neighbourhoods: \t{}".format(m_suburb))
            print("----")

        # X_urban = np.random.randn(m_urban, 2).dot(np.array([[std_feature_1, 0], [0, std_feature_2]]))
        for dim in range(len(std)):
            if dim == 0:
                X_urban = np.random.randn(m_urban, 1) * std[dim]
            else:
                X_urban = np.concatenate((X_urban, np.random.randn(m_urban, 1) * std[dim]), axis=1)

        # X_suburb = np.random.randn(m_suburb, 2).dot(np.array([[std_feature_1, 0], [0, std_feature_2]]))
        for dim in range(len(std)):
            if dim == 0:
                X_suburb = np.random.randn(m_suburb, 1) * std[dim]
            else:
                X_suburb = np.concatenate((X_suburb, np.random.randn(m_suburb, 1) * std[dim]), axis=1)

        # shift suburban population
        shift = np.zeros(len(std))
        shift[0] = 1
        X_suburb += shift

        # sort data points in both sets by feature 1
        X_urban_sorted = sorted(X_urban, key=lambda x: x[1], reverse=False)

        # generate a vector C denoting the racial information
        C = [np.zeros(m_urban, dtype=bool)]

        # number of all non-whites living in urban neighbourhoods
        m0 = int(proportion_0_urban * proportion_0 * number_data_points)
        C[0][m0:] = True

        if self.verbose:
            print("Number of \tnon-white \turban: \t\t{}".format(m0))
            print("Number of \twhite \t\turban: \t\t{}".format(m_urban - m0))

        X_suburb_sorted = sorted(X_suburb, key=lambda x: x[1], reverse=False)
        C.append(np.zeros(m_suburb, dtype=bool))
        # number of all non-whites living in suburban neighbourhoods
        m1 = int(proportion_0_suburban * proportion_0 * number_data_points)
        C[1][m1:] = True

        if self.verbose:
            print("Number of \tnon-white \tsuburban: \t{}".format(m1))
            print("Number of \twhite \t\tsuburban: \t{}".format(m_suburb - m1))
            print("----")

        # generate a vector Y for urban area denoting the actual information about whether
        # money was paid back or not
        Y = [np.zeros(m_urban, dtype=bool)]
        # set
        m00 = int(m0 * proportion_0_dont_pay)
        Y[0][m00:m0] = True
        Y[0][:m0] = shuffle(Y[0][:m0])

        m10 = int((m_urban - m0) * proportion_1_dont_pay)
        Y[0][m0 + m10:] = True
        Y[0][m0:] = shuffle(Y[0][m0:])

        if self.verbose:
            print("Number of \turban \t\tnon-white \tdo not pay\t(m00): {}".format(m00))
            print("Number of \turban \t\twhite \t\tdo not pay\t(m10): {}".format(m10))

        # generate a vector Y for suburban area denoting the actual information about whether
        # money was paid back or not
        Y.append(np.zeros(m_suburb, dtype=bool))
        m00 = int(m1 * proportion_0_dont_pay)
        Y[1][m00:m0] = True
        Y[1][:m0] = shuffle(Y[1][:m0])

        m10 = int((m_suburb - m1) * proportion_1_dont_pay)
        Y[1][m0 + m10:] = True
        Y[1][m0:] = shuffle(Y[1][m0:])

        if self.verbose:
            print("Number of \tsuburban \tnon-white \tdo not pay\t(m00): {}".format(m00))
            print("Number of \tsuburban \twhite \t\tdo not pay\t(m10): {}".format(m10))
            print("----")

        X_full = np.concatenate((X_urban_sorted, X_suburb_sorted))
        C_full = np.concatenate((C[0], C[1]))
        Y_full = np.concatenate((Y[0], Y[1]))

        return X_full, C_full, Y_full


def prepare_plot(X, C, Y, Y_pred=None, prototypes=None, verbose=False, title="Plot", label1="feature 1",
                 label2="feature 2"):
    """
        Generated plot information for a given data set with given classification.

        Parameters
        ----------
        X: np.ndarray of floats
            Array containing all data point coordinates.

        C: np.array of bool
            Array containing all data point subclass memberships.

        Y: np.array of bool
            Array containing all data point outcomes.

        Y_pred: np.array of bool
            Array containing all data point predicted outcomes.

        prototypes: np.ndarray of floats
            Array of prototype locations.

        verbose: bool
                Flag to show status prints.

        Returns
        -------
        ax: subplot info
            Contains prepared plot details.
    """

    color_0 = 'skyblue'
    color_1 = 'scarletred'
    color_pos = 'chocolate'

    # non-white: do not pay
    log00 = np.logical_and(np.logical_not(C), np.logical_not(Y))
    # non-white: pay
    log01 = np.logical_and(np.logical_not(C), Y)

    # white: do not pay
    log10 = np.logical_and(C, np.logical_not(Y))
    # white: pay
    log11 = np.logical_and(C, Y)

    # Plot the data and the prototypes as well
    fig = plt.figure()
    fig.canvas.set_window_title(title)
    ax = fig.add_subplot(111)
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)

    if Y_pred is None:
        # non-white: do not pay
        ax.scatter(X[log00, 0], X[log00, 1], c=_tango_color(color_0, 0),
                   edgecolors=_tango_color(color_0, 2), marker='o')
        # non-white: pay
        ax.scatter(X[log01, 0], X[log01, 1], c=_tango_color(color_0, 0),
                   edgecolors=_tango_color(color_0, 2), marker='s')
        # white: do not pay
        ax.scatter(X[log10, 0], X[log10, 1], c=_tango_color(color_1, 0),
                   edgecolors=_tango_color(color_1, 2), marker='o')
        # white: pay
        ax.scatter(X[log11, 0], X[log11, 1], c=_tango_color(color_1, 0),
                   edgecolors=_tango_color(color_1, 2), marker='s')
    else:
        if verbose:
            n_pos = list(Y_pred).count(True)
            print("Number of positive values in Y_pred: {}".format(n_pos))

        # Gather indices based on classification

        # non-white | do not pay | classified positive
        log00_pos = np.logical_and(log00, Y_pred)
        # non-white | pay | classified positive
        log01_pos = np.logical_and(log01, Y_pred)
        # white | do not pay | classified positive
        log10_pos = np.logical_and(log10, Y_pred)
        # white | pay | classified positive
        log11_pos = np.logical_and(log11, Y_pred)

        if verbose:
            n_pos = list(log00_pos).count(True)
            print("Number of positive values in log01_pos: {} of {}".format(n_pos, len(log01_pos)))

        # non-white | do not pay | classified negative
        log00_neg = np.logical_and(log00, np.logical_not(Y_pred))
        # non-white | pay | classified negative
        log01_neg = np.logical_and(log01, np.logical_not(Y_pred))

        # white| do not pay | classified negative
        log10_neg = np.logical_and(log10, np.logical_not(Y_pred))
        # white | pay  | classified negative
        log11_neg = np.logical_and(log11, np.logical_not(Y_pred))

        if verbose:
            sum_log = list(log00_pos).count(True) + list(log01_pos).count(True) \
                      + list(log10_pos).count(True) + list(log11_pos).count(True) \
                      + list(log00_neg).count(True) + list(log01_neg).count(True) \
                      + list(log10_neg).count(True) + list(log11_neg).count(True)
            print("Sum over all logs is: {}".format(sum_log))

        # non-white: do not pay
        ax.scatter(X[log00_neg, 0], X[log00_neg, 1], c=_tango_color(color_0, 0),
                   edgecolors=_tango_color(color_0, 2), marker='o')
        # non-white: pay
        ax.scatter(X[log01_neg, 0], X[log01_neg, 1], c=_tango_color(color_0, 0),
                   edgecolors=_tango_color(color_0, 2), marker='s')
        # white: do not pay
        ax.scatter(X[log10_neg, 0], X[log10_neg, 1], c=_tango_color(color_1, 0),
                   edgecolors=_tango_color(color_1, 2), marker='o')
        # white: pay
        ax.scatter(X[log11_neg, 0], X[log11_neg, 1], c=_tango_color(color_1, 0),
                   edgecolors=_tango_color(color_1, 2), marker='s')

        # non-white: do not pay
        ax.scatter(X[log00_pos, 0], X[log00_pos, 1], c=_tango_color(color_pos, 0),
                   edgecolors=_tango_color(color_0, 2), marker='o')
        # non-white: pay
        ax.scatter(X[log01_pos, 0], X[log01_pos, 1], c=_tango_color(color_pos, 0),
                   edgecolors=_tango_color(color_0, 2), marker='s')
        # white: do not pay
        ax.scatter(X[log10_pos, 0], X[log10_pos, 1], c=_tango_color(color_pos, 0),
                   edgecolors=_tango_color(color_1, 2), marker='o')
        # white: pay
        ax.scatter(X[log11_pos, 0], X[log11_pos, 1], c=_tango_color(color_pos, 0),
                   edgecolors=_tango_color(color_1, 2), marker='s')
        if prototypes is not None:
            ax.scatter(prototypes[0, 0], prototypes[0, 1], c=_tango_color(color_0, 1),
                       edgecolors=_tango_color(color_0, 2), linewidths=2, s=150, marker='D')
            ax.scatter(prototypes[1, 0], prototypes[1, 1], c=_tango_color(color_1, 1),
                       edgecolors=_tango_color(color_1, 2), linewidths=2, s=150, marker='D')
    if verbose:
        print("----")
    return ax


def plot_prepared_dist(ax):
    """
        Plots a prepared data set.

        Parameters
        ----------
        ax: axes
            Axes object that holds the plotting information.
    """
    ax.plot()
    plt.show()
    return


def plot_prepared_dist_multi(ax_list):
    """
        Plots a prepared data set.

        Parameters
        ----------
        ax: axes
            Axes object that holds the plotting information.
    """
    for ax in ax_list:
        ax.plot()
    plt.show()
    return


def plot_dist(X, C, Y, Y_pred, prototypes=None, verbose=False):
    """
        Prints classification of a given data set.

        Parameters
        ----------
        X: np.ndarray of floats
            Array containing all data point coordinates.

        C: np.array of bool
            Array containing all data point subclass memberships.

        Y: np.array of bool
            Array containing all data point outcomes.

        Y_pred: np.array of bool
            Array containing all data point predicted outcomes.

        prototypes: np.ndarray of floats
            Array of prototype locations.

        verbose: bool
                Flag to show status prints.

    """
    ax = prepare_plot(X, C, Y, Y_pred, prototypes, verbose=False)
    plot_prepared_dist(ax, verbose)
    return


def print_dist_glvq(X, C, Y, verbose=False):
    """
        Prints classification of a given data set with GLVQ.

        Parameters
        ----------
        X: np.ndarray of floats
            Array containing all data point coordinates.

        C: np.array of bool
            Array containing all data point subclass memberships.

        Y: np.array of bool
            Array containing all data point outcomes.

        verbose: bool
                Flag to show status prints.
    """

    # Train a GLVQ model
    model = GlvqModel()
    model.fit(X, Y)

    # find prediction of model for all points
    Y_pred = model.predict(X)

    if verbose:
        n_pos = list(Y_pred).count(True)
        print("Number of positive values in Y_pred: {}".format(n_pos))
        print("----")

    plot_dist(X, C, Y, Y_pred, model.w_)
    return


def normalize_feature(feature):
    """
        Normalizes a feature by l2 norm.

        Parameters
        ----------
        feature: (n x 1) np.array of floats
            Unnormalized feature vector.

        Returns
        -------
        normalized_feature: (n x 1) np.array of floats
            Normalized feature vector.
    """
    normalized_feature = normalize(feature.T).T
    return normalized_feature


# TODO: Check that True == 1
def normalize_binary_feature(feature):
    """
        Normalizes a feature by l2 norm.

        Parameters
        ----------
        feature: (n x 1) np.array of floats
            Binary feature vector.

        Returns
        -------
        normalized_feature: (n x 1) np.array of floats
            Normalized feature vector.
    """
    n, m = feature.shape
    values = np.unique(feature)
    normalized_feature = np.zeros((n, m))
    for i in range(m):
        if feature[0, i] == values[1]:
            normalized_feature[0, i] = 1
    return normalized_feature


# WARNING: number of values is computed and not a parameter. Could lead to problems if one value is not in set.
def normalize_category_feature(feature):
    """
        Normalizes a feature by l2 norm.

        Parameters
        ----------
        feature: (n x 1) np.array of floats
            Un-normalized feature vector.

        Returns
        -------
        normalized_feature: (n x 1) np.array of floats
            Normalized feature vector.
    """
    _, n = feature.shape
    # print("n: {}".format(n))

    values = np.unique(feature)

    vertices = equilateral_simplex(len(values))

    # check_dist(vertices)

    # dimension of each vertex
    _, m = vertices.shape

    normalized_feature = np.zeros((n, m))
    for i in range(n):
        # find index of i-th element of feature in values

        j = list(values).index(feature.T[i])

        # place vertex of that index in normalized vector
        normalized_feature[i] = vertices[j]

    # print(normalized_feature)
    return normalized_feature.T


def equilateral_simplex(n, verbose=False):
    """
        Normalizes a feature by l2 norm.

        Parameters
        ----------
        n: int
            Number of dimensions.

        verbose: bool
                Flag to show status prints.

        Returns
        -------
        vertices: np.array of floats
            Vector of coordinates of the vertices.
    """

    if verbose:
        print("equilateral_simplex:")

    # Initialise X with n x (n-1) zeroes
    vertices = np.zeros((n, (n - 1)))
    # Initialise p with n x 1 zeroes
    p = np.zeros(n)

    for i in range(1, n):
        if verbose:
            print("i: {}".format(i))
        for j in range(0, (i - 1)):
            if verbose:
                print("i: {}\tj: {}\tp[j]: {}".format(i, j, p[j]))
            vertices[i][j] = p[j + 1]
        p[i] = 1 / math.sqrt(2 * (i + 1) * i)
        if verbose:
            print("p{}: {}".format(i, p[i]))
        vertices[i, (i - 1)] = p[i] * (i + 1)

    if verbose:
        print("----")

    return vertices


def check_dist(vertices):
    """
        Helper function to check whether the distance between point in a vector.

        Parameters
        ----------
        vertices: np.array of floats
            Vector of coordinates of the vertices.
    """
    for v in vertices:
        print(v)
        for u in vertices:
            if not (v == u).all():
                print("dist of {}\t and \t{}\t: \t{}".format(v, u, np.linalg.norm(v - u)))

    return


def normalize_metric_feature(feature):
    """
        Normalizes a feature by l2 norm.

        Parameters
        ----------
        feature: (n x 1) np.array of floats
            Metric feature vector.

        Returns
        -------
        normalized_feature: (n x 1) np.array of floats
            Normalized feature vector.
    """
    max_val = 0
    min_val = np.inf

    print("type(feature): {}".format(type(feature)))
    print("feature.shape: {}".format(feature.shape))
    print("feature.T[0]: {}".format(feature.T[0]))

    n, m = feature.shape

    normalized_feature = np.zeros((n, m))

    for i in range(m):

        normalized_feature[0][i] = feature[0][i].astype(np.float)

        val = normalized_feature[0][i]

        if val > max_val:
            max_val = val
        elif val < min_val:
            min_val = val

    dist = max_val - min_val

    normalized_feature = normalized_feature / dist

    return normalized_feature
