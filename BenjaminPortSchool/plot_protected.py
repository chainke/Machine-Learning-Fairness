# Plotting Function for Fairness Experiments
#
# (c) 2018 AG ML
# CITEC Center of Excellence
# Bielefeld University
from GLVQ.plot_2d import tango_color
import matplotlib.pyplot as plt
import numpy as np


def plot_protected(X, Y, Y_pred=None, Z=None, W=None, Y_W=None, fig=None):
    # Plot a data set with two dimensions (!), such that labels
    # are indicated by fill color, predicted labels are indicated
    # by edge color, and protected group membership is indicated
    # by shape. Optionally, this function can also plot prototypes.
    # Let n be the number of data points and K be the number of
    # prototypes. Then, this method expects the following input:
    # X - a n x 2 numpy array/matrix containing the data locations
    # Y - a n x 1 numpy array containing the true labels for each
    #     data point, either 0 or 1.
    # Y_pred - a n x 1 numpy array containing the predicted labels
    #     for each data point in the range [0, 1]
    # Z - a n x 1 numpy array containing the protection status of
    #     each data point, either 0 (non-protected), or 1
    #     (protected).
    # W - a K x 2 numpy array/matrix containing the prototype
    #     locations.
    # Y_W - a K x 1 numpy array containing the labels for the
    #     prototypes, either 0 or 1.
    # fig - a pre-defined figure handle to plot to

    m = X.shape[0]

    # Pre-process input
    if (Y_pred is None):
        Y_pred = Y
    if (Z is None):
        Z = np.zeros(m)

    if (fig is None):
        fig = plt.figure()

    ax = fig.add_subplot(111)

    # Plot non-protected people who have a negative label and are
    # predicted as zero label
    selec = np.logical_and(Z == 0, np.logical_and(Y == 0, Y_pred < 0.5))
    ax.scatter(X[selec, 0], X[selec, 1], c=tango_color('skyblue', 0), edgecolors=tango_color('skyblue', 2), marker='o')
    # Plot non-protected people who have a negative label but are
    # predicted as positive label
    selec = np.logical_and(Z == 0, np.logical_and(Y == 0, Y_pred >= 0.5))
    ax.scatter(X[selec, 0], X[selec, 1], c=tango_color('skyblue', 0), edgecolors=tango_color('scarletred', 2),
               marker='o')
    # Plot non-protected people who have a positive label but are
    # predicted as negative label
    selec = np.logical_and(Z == 0, np.logical_and(Y == 1, Y_pred < 0.5))
    ax.scatter(X[selec, 0], X[selec, 1], c=tango_color('scarletred', 0), edgecolors=tango_color('skyblue', 2),
               marker='o')
    # Plot non-protected people who have a positive label and are
    # predicted as positive label
    selec = np.logical_and(Z == 0, np.logical_and(Y == 1, Y_pred >= 0.5))
    ax.scatter(X[selec, 0], X[selec, 1], c=tango_color('scarletred', 0), edgecolors=tango_color('scarletred', 2),
               marker='o')

    # Plot protected people who have a negative label and are
    # predicted as zero label
    selec = np.logical_and(Z == 1, np.logical_and(Y == 0, Y_pred < 0.5))
    ax.scatter(X[selec, 0], X[selec, 1], c=tango_color('skyblue', 0), edgecolors=tango_color('skyblue', 2), marker='s')
    # Plot protected people who have a negative label but are
    # predicted as positive label
    selec = np.logical_and(Z == 1, np.logical_and(Y == 0, Y_pred >= 0.5))
    ax.scatter(X[selec, 0], X[selec, 1], c=tango_color('skyblue', 0), edgecolors=tango_color('scarletred', 2),
               marker='s')
    # Plot protected people who have a positive label but are
    # predicted as negative label
    selec = np.logical_and(Z == 1, np.logical_and(Y == 1, Y_pred < 0.5))
    ax.scatter(X[selec, 0], X[selec, 1], c=tango_color('scarletred', 0), edgecolors=tango_color('skyblue', 2),
               marker='s')
    # Plot protected people who have a positive label and are
    # predicted as positive label
    selec = np.logical_and(Z == 1, np.logical_and(Y == 1, Y_pred >= 0.5))
    ax.scatter(X[selec, 0], X[selec, 1], c=tango_color('scarletred', 0), edgecolors=tango_color('scarletred', 2),
               marker='s')

    # Plot prototypes
    if (W is not None):
        # Plot prototypes for the negative label
        ax.scatter(W[Y_W == 0, 0], W[Y_W == 0, 1], c=tango_color('skyblue', 1), edgecolors=tango_color('skyblue', 2),
                   linewidths=2, s=150, marker='D')
        # Plot prototypes for the positive label
        ax.scatter(W[Y_W == 1, 0], W[Y_W == 1, 1], c=tango_color('scarletred', 1),
                   edgecolors=tango_color('scarletred', 2), linewidths=2, s=150, marker='D')
