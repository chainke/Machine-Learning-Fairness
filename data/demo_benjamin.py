# Fair GLVQ Demo
#
# (c) 2018 AG ML
# CITEC Center of Excellence
# Bielefeld University

import math
import numpy as np
import matplotlib.pyplot as plt
from GLVQ.plot_2d import tango_color
from GLVQ.plot_2d import to_tango_colors
from GLVQ.plot_2d import plot2d
from GLVQ.glvq import GlvqModel
from fair_glvq import GlvqModel as FairGlvqModel

# Assume we analyze the credit scoring algorithm of a bank. The credit scoring algorithm
# has the purpose to predict how likely it is that customers will pay back their debt.
# Formally, the scoring algorithm should be a function f which maps a vector of observable
# features x to a credit score y in the range [0, 1], where f(x) = 1 if the algorithm is
# absolutely certain that the customer will pay their money back and f(x) = 0 if the algorithm
# is absolutely certain that the customer will _not_ pay their money back.
#
# In our example, let us assume that the observable features are the annual income and the
# place of residence of a customer, parametrized by the distance to the city center. The
# reasoning goes that people with a higher annual income are more likely to pay back their
# money and that people who live in the suburbs (higher distance from the center) are more
# likely to pay back their money.
#
# To learn the scoring algorithm the bank records training data from their customers, complete
# with annual income, distance from city center, and whether they actually did pay their
# money back or not (y = 1 or y = 0). We also assume that the bank records racial information
# to detect discriminatory behavior of their scoring algorithm. In particular, the bank records
# whether a customer is white (c = 1) or non-white (c = 0).
#
# The recorded training data may look something like this:

# Number of training data points
m = 1000
# Gaussian standard deviation for distance from city center
std_1 = 0.2
# Gaussian standard deviation for income
std_2 = 0.2

# proportion of non-white people who do not pay their money back (if 0.5 non-discriminatory
# classification is trivial)
p0 = 0.8
# proportion of white people who do not pay their money back
p1 = 0.5
# proportion of non-white people in the overall data set
q = 0.5

# generate a vector C denoting the racial information
C = np.zeros(m, dtype=bool)
m0 = int(m * q)
C[m0:] = True
# generate a vector Y denoting the actual information about whether
# money was paid back or not
Y = np.zeros(m, dtype=bool)
m00 = int(m0 * p0)
Y[m00:m0] = True
m10 = int((m - m0) * p1)
Y[m0 + m10:] = True

# generate normally distributed base data
X = np.random.randn(m, 2).dot(np.array([[std_1, 0], [0, std_2]]))

# shift data for non-white people who do not pay their money back
log00 = np.logical_and(np.logical_not(C), np.logical_not(Y))
X[log00, :] += np.array([0, 0])
# shift data for non-white people who do pay their money back
log01 = np.logical_and(np.logical_not(C), Y)
X[log01, :] += np.array([1, 1])
# shift data for white people who do not pay their money back
log10 = np.logical_and(C, np.logical_not(Y))
X[log10, :] += np.array([0, 1])
# shift data for white people who do pay their money back
log11 = np.logical_and(C, Y)
X[log11, :] += np.array([1, 2])

# print(C)
# print(Y)

# Train a GLVQ model
model = GlvqModel()
model.fit(X, Y)

# Check some fairness measures
Y_predicted = model.predict(X)
# Compute the mean difference, that is, the difference between the average credit score for
# whites and non-whites
# First, we need the credit score for that, which is _not_ the classification, but the function
# sigma( (d[0] - d[1]) / (d[0] + d[1]) ) where d[0] is the distance of the data point to the
# prototype for people who will pay their money back and d[1] is the distance of the data point
# to the prototype for people who will _not_ pay their money back.
D = model._compute_distance(X)
f = np.divide(D[:, 0] - D[:, 1], D[:, 0] + D[:, 1])
f = np.divide(np.ones(m), 1 + np.exp(-f))

print('mean difference: {}'.format(np.mean(f[C]) - np.mean(f[np.logical_not(C)])))

# Compute predictive equality measurements, that is, the fraction of people in a protected group
# who are erroneously classified
print('\npredictive equality measurements:')
print('fraction of non-whites who would not pay their money back but get a good score: {}'.format(
    np.mean(np.not_equal(Y_predicted[log00], Y[log00]))))
print('fraction of non-whites who would pay their money back but get a bad score: {}'.format(
    np.mean(np.not_equal(Y_predicted[log01], Y[log01]))))
print('fraction of whites who would not pay their money back but get a good score: {}'.format(
    np.mean(np.not_equal(Y_predicted[log10], Y[log10]))))
print('fraction of whites who would pay their money back but get a bad score: {}'.format(
    np.mean(np.not_equal(Y_predicted[log11], Y[log11]))))


def getData():
    return X, Y, Y_predicted


def getProtected():
    protected = []

    for i in range(len(C)):
        if (C[i]):
            protected.append(0)
        else:
            protected.append(1)

    # print(protected)

    return protected


def getTrainedModel():
    return model


protected_label = getProtected()
fair_model = FairGlvqModel(100)
fair_model.fit(X, Y, protected_label)

# Check some fairness measures
fair_Y_predicted = fair_model.predict(X)

counter = 0
for i in range(len(Y_predicted)):
    if fair_Y_predicted[i] != Y_predicted[i]:
        print("fair: " + str(fair_Y_predicted[i]))
        print("normal: " + str(Y_predicted[i]))
        counter = counter+1

print("number of different outcomes: " + str(counter))



# Compute the mean difference, that is, the difference between the average credit score for
# whites and non-whites
# First, we need the credit score for that, which is _not_ the classification, but the function
# sigma( (d[0] - d[1]) / (d[0] + d[1]) ) where d[0] is the distance of the data point to the
# prototype for people who will pay their money back and d[1] is the distance of the data point
# to the prototype for people who will _not_ pay their money back.
fair_D = fair_model._compute_distance(X)
ff = np.divide(fair_D[:, 0] - D[:, 1], fair_D[:, 0] + fair_D[:, 1])
ff = np.divide(np.ones(m), 1 + np.exp(-ff))

print('mean difference: {}'.format(np.mean(ff[C]) - np.mean(ff[np.logical_not(C)])))

# Compute predictive equality measurements, that is, the fraction of people in a protected group
# who are erroneously classified
print('\npredictive equality measurements:')
print('fraction of non-whites who would not pay their money back but get a good score: {}'.format(
    np.mean(np.not_equal(fair_Y_predicted[log00], Y[log00]))))
print('fraction of non-whites who would pay their money back but get a bad score: {}'.format(
    np.mean(np.not_equal(fair_Y_predicted[log01], Y[log01]))))
print('fraction of whites who would not pay their money back but get a good score: {}'.format(
    np.mean(np.not_equal(fair_Y_predicted[log10], Y[log10]))))
print('fraction of whites who would pay their money back but get a bad score: {}'.format(
    np.mean(np.not_equal(fair_Y_predicted[log11], Y[log11]))))

# ax1  = fig.add_subplot(111)
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# Plot the data and the prototypes as well
f.canvas.set_window_title("LVQ with continuous distance to city center")
ax1.set_xlabel("Distance from City Center")
ax1.set_ylabel("Income")
ax1.scatter(X[log00, 0], X[log00, 1], c=tango_color('skyblue', 0), edgecolors=tango_color('skyblue', 2), marker='o')
ax1.scatter(X[log01, 0], X[log01, 1], c=tango_color('scarletred', 0), edgecolors=tango_color('scarletred', 2),
            marker='o')
ax1.scatter(X[log10, 0], X[log10, 1], c=tango_color('skyblue', 0), edgecolors=tango_color('skyblue', 2), marker='s')
ax1.scatter(X[log11, 0], X[log11, 1], c=tango_color('scarletred', 0), edgecolors=tango_color('scarletred', 2),
            marker='s')
ax1.scatter(model.w_[0, 0], model.w_[0, 1], c=tango_color('skyblue', 1), edgecolors=tango_color('skyblue', 2),
            linewidths=2, s=150, marker='D')
ax1.scatter(model.w_[1, 0], model.w_[1, 1], c=tango_color('scarletred', 1), edgecolors=tango_color('scarletred', 2),
            linewidths=2, s=150, marker='D')

ax2.set_xlabel("Distance from City Center")
ax2.set_ylabel("Income")
ax2.scatter(X[log00, 0], X[log00, 1], c=tango_color('skyblue', 0), edgecolors=tango_color('skyblue', 2), marker='o')
ax2.scatter(X[log01, 0], X[log01, 1], c=tango_color('scarletred', 0), edgecolors=tango_color('scarletred', 2),
            marker='o')
ax2.scatter(X[log10, 0], X[log10, 1], c=tango_color('skyblue', 0), edgecolors=tango_color('skyblue', 2), marker='s')
ax2.scatter(X[log11, 0], X[log11, 1], c=tango_color('scarletred', 0), edgecolors=tango_color('scarletred', 2),
            marker='s')
ax2.scatter(fair_model.w_[0, 0], fair_model.w_[0, 1], c=tango_color('skyblue', 1), edgecolors=tango_color('skyblue', 2),
            linewidths=2, s=150, marker='D')
ax2.scatter(fair_model.w_[1, 0], fair_model.w_[1, 1], c=tango_color('scarletred', 1),
            edgecolors=tango_color('scarletred', 2), linewidths=2, s=150, marker='D')
#plt.show()
