# Student Experiment
#
# (c) 2018 AG ML
# CITEC Center of Excellence
# Bielefeld University

import math
import numpy as np
import matplotlib.pyplot as plt
import csv
# from glvq.plot_2d import tango_color
# from glvq.plot_2d import to_tango_colors
# from glvq.plot_2d import plot2d
from GLVQ.glvq import GlvqModel
import quad_fair_glvq as quad_glvq
import abs_fair_glvq as abs_glvq
from BenjaminPortSchool.plot_protected import plot_protected

# Load student CSV data. Data set is available at
# http://archive.ics.uci.edu/ml/datasets/Student+Performance

# The feature matrix
X = []
# The label we wish to predict (the grade)
Y = []
# The protected attribute
Z = []
with open('../data/uci_student/student-por.csv', newline='') as barriers_positions_file:
    reader = csv.reader(barriers_positions_file, delimiter=';')
    # Drop the column headers
    next(reader)
    # Read all data in the CSV data, row by row
    for data_row in reader:
        # The features for the current data point
        # Note that we only read the grade data
        # here; so we try to predict the final
        # grade from the grade in the first and
        # second period
        x = np.zeros(2)
        #		x = np.zeros(10)
        # first period grade
        x[0] = float(data_row[30])
        # second period grade
        x[1] = float(data_row[31])
        #		# traveltime
        #		x[2] = float(data_row[12])
        #		# studytime
        #		x[3] = float(data_row[13])
        #		# no. of failures
        #		x[4] = float(data_row[14])
        #		# school support
        #		x[5] = 1 if data_row[15] == 'yes' else 0
        #		# family support
        #		x[6] = 1 if data_row[16] == 'yes' else 0
        #		# internet
        #		x[7] = 1 if data_row[21] == 'yes' else 0
        #		# freetime
        #		x[8] = float(data_row[24])
        #		# absences
        #		x[9] = float(data_row[29])

        X.append(x)

        # The label for the current data point (third period grade),
        # converted to binary (above 10 or below 10)
        Y.append(1 if int(data_row[32]) > 10 else 0)

        # The protected attribute for the current data point
        # Internet
        Z.append(1 if data_row[21] == 'yes' else 0)
    # age
# Z.append(1 if int(data_row[2]) > 17 else 0)

# convert python lists to numpy arrays
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

# Print the empiric estimate of P(Z)
print('{} percent of people are in the protected group.'.format(np.mean(Z == 1)))

# Print the empiric estimates of P(Y|Z) for
# Z = 0 and Z = 1. If these probabilities are about equal,
# there is likely no fairness problem, because accurate classification
# is possible without affecting fairness.
print('{} percent of people in the protected group have a good grade.'.format(
    np.sum(np.logical_and(Z == 1, Y == 1)) / np.sum(Z == 1)))
print('{} percent of people outside the protected group have a good grade.'.format(
    np.sum(np.logical_and(Z == 0, Y == 1)) / np.sum(Z == 0)))

# normalize the feature space by dividing by the maximum value in each column
X = np.divide(X, np.amax(X, axis=0))

# try to classify the data using Glvq
model = GlvqModel()
model.fit(X, Y)
Y_pred = model.predict(X)

print('classification accuracy:', model.score(X, Y))

# Plot the data and the prototypes as well
plot_protected(X, Y, Y_pred, Z, model.w_, model.c_w_)

plt.show()

# Compute the mean difference, that is, the difference in confidence
# for a good grade between non-members versus members of the protected group.
# First, we need the confidence measure of GLVQ for hat purpose, which is
# sigma( (d[0] - d[1]) / (d[0] + d[1]) ) where d[0] is the distance of the
# data point to the prototype for bad grades and d[1] is the distance of the
# data point to the prototype for good grades, and sigma is some nonlinear
# function, in this case the logistic function 1 / (1 + exp(-x))
D = model._compute_distance(X)
f = np.divide(D[:, 0] - D[:, 1], D[:, 0] + D[:, 1])
f = np.divide(np.ones(Y.shape[0]), 1 + np.exp(-f))

print('mean difference: {}'.format(np.mean(f[Z == 0]) - np.mean(f[Z == 1])))


#------------------------------

set_alpha = 0.11

# same thing but with a fair classifier
model = abs_glvq.MeanDiffGlvqModel(alpha=set_alpha)
model.fit_fair(X, Y, Y)
Y_pred = model.predict(X)

print('classification accuracy:', model.score(X, Y))

# Plot the data and the prototypes as well
plot_protected(X, Y, Y_pred, Z, model.w_, model.c_w_)

# Compute the mean difference, that is, the difference in confidence
# for a good grade between non-members versus members of the protected group.
# First, we need the confidence measure of GLVQ for hat purpose, which is
# sigma( (d[0] - d[1]) / (d[0] + d[1]) ) where d[0] is the distance of the
# data point to the prototype for bad grades and d[1] is the distance of the
# data point to the prototype for good grades, and sigma is some nonlinear
# function, in this case the logistic function 1 / (1 + exp(-x))
D = model._compute_distance(X)
f = np.divide(D[:, 0] - D[:, 1], D[:, 0] + D[:, 1])
f = np.divide(np.ones(Y.shape[0]), 1 + np.exp(-f))

print('mean difference: {}'.format(np.mean(f[Z == 0]) - np.mean(f[Z == 1])))

plt.show()
