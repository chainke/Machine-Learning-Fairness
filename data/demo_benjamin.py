# Fair GLVQ Demo
#
# (c) 2018 AG ML
# CITEC Center of Excellence
# Bielefeld University

import math
import numpy as np
import matplotlib.pyplot as plt
from GLVQ.plot_2d import tango_color
from sklearn_lvq.glvq import GlvqModel
from abs_fair_glvq import MeanDiffGlvqModel as absolutFairGlvqModel
from quad_fair_glvq import MeanDiffGlvqModel as quadraticFairGlvqModel
from normalized_fair_glvq import NormMeanDiffGlvqModel as NormFairGlvqModel
from matplotlib import cm


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
q = 0.2

# fairness factor
alpha1 = 200
alpha2 = 400
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

minimum = min(np.mean(f)/q, (1-np.mean(f))/(1-q))
print('mean difference: {}'.format((np.mean(f[C]) - np.mean(f[np.logical_not(C)]))))
print('normalized mean difference: {}'.format((np.mean(f[C]) - np.mean(f[np.logical_not(C)]))/minimum))

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
print('accuracy of the classifier:   ' + str(model.score(X, Y)))


def getData():
    return X, Y, Y_predicted


def getProtected():
    protected = []

    for i in range(len(C)):
        if (C[i]):
            protected.append(0)
        else:
            protected.append(1)

    return protected


def getTrainedModel():
    return model


protected_label = getProtected()

# Normalized Mean Difference
fair_model = NormFairGlvqModel(alpha1)
fair_model.fit_fair(X, Y, protected_label)

norm_fair_model = NormFairGlvqModel(alpha2)
norm_fair_model.fit_fair(X, Y, protected_label)


# Check some fairness measures
fair_Y_predicted = fair_model.predict(X)

# Check some fairness measures
norm_fair_Y_predicted = norm_fair_model.predict(X)

# Compute the mean difference, that is, the difference between the average credit score for
# whites and non-whites
# First, we need the credit score for that, which is _not_ the classification, but the function
# sigma( (d[0] - d[1]) / (d[0] + d[1]) ) where d[0] is the distance of the data point to the
# prototype for people who will pay their money back and d[1] is the distance of the data point
# to the prototype for people who will _not_ pay their money back.
fair_D = fair_model._compute_distance(X)
ff = np.divide(fair_D[:, 0] - D[:, 1], fair_D[:, 0] + fair_D[:, 1])
ff = np.divide(np.ones(m), 1 + np.exp(-ff))

norm_fair_D = norm_fair_model._compute_distance(X)
fff = np.divide(norm_fair_D[:, 0] - D[:, 1], norm_fair_D[:, 0] + norm_fair_D[:, 1])
fff = np.divide(np.ones(m), 1 + np.exp(-fff))


minimum = min(np.mean(ff)/q, (1-np.mean(ff))/(1-q))
print('[fair] alpha1 normalized mean difference: {}'.format((np.mean(ff[C]) - np.mean(ff[np.logical_not(C)]))/minimum))

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
print('accuracy of the classifier:   ' + str(fair_model.score(X, Y)))

print(' ')
minimum = min(np.mean(fff)/q, (1-np.mean(fff))/(1-q))
print('[fair] alpha 2 normalized mean difference: {}'.format((np.mean(fff[C]) - np.mean(fff[np.logical_not(C)]))/minimum))

# Compute predictive equality measurements, that is, the fraction of people in a protected group
# who are erroneously classified
print('\npredictive equality measurements:')
print('fraction of non-whites who would not pay their money back but get a good score: {}'.format(
    np.mean(np.not_equal(norm_fair_Y_predicted[log00], Y[log00]))))
print('fraction of non-whites who would pay their money back but get a bad score: {}'.format(
    np.mean(np.not_equal(norm_fair_Y_predicted[log01], Y[log01]))))
print('fraction of whites who would not pay their money back but get a good score: {}'.format(
    np.mean(np.not_equal(norm_fair_Y_predicted[log10], Y[log10]))))
print('fraction of whites who would pay their money back but get a bad score: {}'.format(
    np.mean(np.not_equal(norm_fair_Y_predicted[log11], Y[log11]))))
print('accuracy of the classifier:   ' + str(norm_fair_model.score(X, Y)))
print(' ')
# ax1  = fig.add_subplot(111)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

h = .02  # step size in the mesh

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim(yy.min(), yy.max())

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax1.contourf(xx, yy, Z, colors = ('#C9FFD2', '#000000', '#FFF5C9', 'w'), alpha=.8)

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

ax2.set_xlim(xx.min(), xx.max())
ax2.set_ylim(yy.min(), yy.max())

Z = fair_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax2.contourf(xx, yy, Z, colors = ('#C9FFD2', '#000000', '#FFF5C9', 'w'), alpha=.8)

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

ax3.set_xlim(xx.min(), xx.max())
ax3.set_ylim(yy.min(), yy.max())

Z = norm_fair_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax3.contourf(xx, yy, Z, colors = ('#C9FFD2', '#000000', '#FFF5C9', 'w'), alpha=.8)


# Plot the data and the prototypes as well
f.canvas.set_window_title("LVQ Normalized Mean Difference")
ax1.set_xlabel("Income")
ax1.set_ylabel("Distance from City Center")
ax1.set_title('alpha = 0')
ax1.scatter(X[log00, 0], X[log00, 1], c='#2c7a5d', edgecolors='#10553c', marker='o', label="protected, y = 0")
ax1.scatter(X[log01, 0], X[log01, 1], c='#CC6600', edgecolors='#95682D', marker='o', label="protected, y = 1")
ax1.scatter(X[log10, 0], X[log10, 1], c='#00cc00', edgecolors='#006600', marker='s', label="not protected, y = 0")
ax1.scatter(X[log11, 0], X[log11, 1], c='#fc9f00', edgecolors='#E9890A', marker='s',label="not protected, y = 1")
ax1.scatter(model.w_[0, 0], model.w_[0, 1], c='#2c7a5d', edgecolors='#10553c',
            linewidths=2, s=150, marker='D', label="prototype, z = 0")
ax1.scatter(model.w_[1, 0], model.w_[1, 1], c='#CC6600', edgecolors='#95682D',
            linewidths=2, s=150, marker='D',label="prototype, z = 1")
ax1.legend(loc = 2)

ax2.set_xlabel("Income")
ax2.set_ylabel("Distance from City Center")
ax2.scatter(X[log00, 0], X[log00, 1], c='#2c7a5d', edgecolors='#10553c', marker='o')
ax2.scatter(X[log01, 0], X[log01, 1], c='#CC6600', edgecolors='#95682D',
           marker='o')
ax2.scatter(X[log10, 0], X[log10, 1], c='#00cc00', edgecolors='#006600', marker='s')
ax2.scatter(X[log11, 0], X[log11, 1], c='#fc9f00', edgecolors='#E9890A',
           marker='s')
ax2.scatter(fair_model.w_[0, 0], fair_model.w_[0, 1], c='#2c7a5d', edgecolors='#10553c',
           linewidths=2, s=150, marker='D')
ax2.scatter(fair_model.w_[1, 0], fair_model.w_[1, 1], c='#CC6600', edgecolors='#95682D', linewidths=2, s=150, marker='D')
ax2.set_title('alpha = 200')
ax3.set_xlabel("Income")
ax3.set_ylabel("Distance from City Center")
ax3.scatter(X[log00, 0], X[log00, 1], c='#2c7a5d', edgecolors='#10553c', marker='o')
ax3.scatter(X[log01, 0], X[log01, 1], c='#CC6600', edgecolors='#95682D',
            marker='o')
ax3.scatter(X[log10, 0], X[log10, 1], c='#00cc00', edgecolors='#006600', marker='s')
ax3.scatter(X[log11, 0], X[log11, 1],  c='#fc9f00', edgecolors='#E9890A',
            marker='s')
ax3.scatter(norm_fair_model.w_[0, 0], norm_fair_model.w_[0, 1],  c='#2c7a5d', edgecolors='#10553c',
            linewidths=2, s=150, marker='D')
ax3.scatter(norm_fair_model.w_[1, 0], norm_fair_model.w_[1, 1], c='#CC6600', edgecolors='#95682D', linewidths=2, s=150, marker='D')
ax3.set_title('alpha = 400')
f.set_size_inches(18.5,10.5)
f.savefig('./evaluation/05ratio.eps', format='eps')
plt.show()

#


# Mean Difference
fair_model = quadraticFairGlvqModel(alpha1)
fair_model.fit_fair(X, Y, protected_label)

norm_fair_model = quadraticFairGlvqModel(alpha2)
norm_fair_model.fit_fair(X, Y, protected_label)

# Check some fairness measures
fair_Y_predicted = fair_model.predict(X)

# Check some fairness measures
norm_fair_Y_predicted = norm_fair_model.predict(X)

# Compute the mean difference, that is, the difference between the average credit score for
# whites and non-whites
# First, we need the credit score for that, which is _not_ the classification, but the function
# sigma( (d[0] - d[1]) / (d[0] + d[1]) ) where d[0] is the distance of the data point to the
# prototype for people who will pay their money back and d[1] is the distance of the data point
# to the prototype for people who will _not_ pay their money back.
fair_D = fair_model._compute_distance(X)
ff = np.divide(fair_D[:, 0] - D[:, 1], fair_D[:, 0] + fair_D[:, 1])
ff = np.divide(np.ones(m), 1 + np.exp(-ff))

norm_fair_D = norm_fair_model._compute_distance(X)
fff = np.divide(norm_fair_D[:, 0] - D[:, 1], norm_fair_D[:, 0] + norm_fair_D[:, 1])
fff = np.divide(np.ones(m), 1 + np.exp(-fff))





print('[fair] alpha1  mean difference: {}'.format((np.mean(ff[C]) - np.mean(ff[np.logical_not(C)]))))

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
print('accuracy of the classifier:   ' + str(fair_model.score(X, Y)))

print(' ')
print('[fair] alpha 2  mean difference: {}'.format((np.mean(fff[C]) - np.mean(fff[np.logical_not(C)]))))

# Compute predictive equality measurements, that is, the fraction of people in a protected group
# who are erroneously classified
print('\npredictive equality measurements:')
print('fraction of non-whites who would not pay their money back but get a good score: {}'.format(
    np.mean(np.not_equal(norm_fair_Y_predicted[log00], Y[log00]))))
print('fraction of non-whites who would pay their money back but get a bad score: {}'.format(
    np.mean(np.not_equal(norm_fair_Y_predicted[log01], Y[log01]))))
print('fraction of whites who would not pay their money back but get a good score: {}'.format(
    np.mean(np.not_equal(norm_fair_Y_predicted[log10], Y[log10]))))
print('fraction of whites who would pay their money back but get a bad score: {}'.format(
    np.mean(np.not_equal(norm_fair_Y_predicted[log11], Y[log11]))))
print('accuracy of the classifier:   ' + str(norm_fair_model.score(X, Y)))
# ax1  = fig.add_subplot(111)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

h = .02  # step size in the mesh

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

ax1.set_xlim(xx.min(), xx.max())
ax1.set_ylim(yy.min(), yy.max())

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax1.contourf(xx, yy, Z, colors = ('#C9FFD2', '#000000', '#FFF5C9', 'w'), alpha=.8)

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

ax2.set_xlim(xx.min(), xx.max())
ax2.set_ylim(yy.min(), yy.max())

Z = fair_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax2.contourf(xx, yy, Z, colors = ('#C9FFD2', '#000000', '#FFF5C9', 'w'), alpha=.8)

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

ax3.set_xlim(xx.min(), xx.max())
ax3.set_ylim(yy.min(), yy.max())

Z = norm_fair_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax3.contourf(xx, yy, Z, colors = ('#C9FFD2', '#000000', '#FFF5C9', 'w'), alpha=.8)


# Plot the data and the prototypes as well
f.canvas.set_window_title("LVQ Normalized Mean Difference")
ax1.set_xlabel("Income")
ax1.set_ylabel("Distance from City Center")
ax1.set_title('alpha = 0')
ax1.scatter(X[log00, 0], X[log00, 1], c='#2c7a5d', edgecolors='#10553c', marker='o', label="protected, y = 0")
ax1.scatter(X[log01, 0], X[log01, 1], c='#CC6600', edgecolors='#95682D', marker='o', label="protected, y = 1")
ax1.scatter(X[log10, 0], X[log10, 1], c='#00cc00', edgecolors='#006600', marker='s', label="not protected, y = 0")
ax1.scatter(X[log11, 0], X[log11, 1], c='#fc9f00', edgecolors='#E9890A', marker='s',label="not protected, y = 1")
ax1.scatter(model.w_[0, 0], model.w_[0, 1], c='#2c7a5d', edgecolors='#10553c',
            linewidths=2, s=150, marker='D', label="prototype, z = 0")
ax1.scatter(model.w_[1, 0], model.w_[1, 1], c='#CC6600', edgecolors='#95682D',
            linewidths=2, s=150, marker='D',label="prototype, z = 1")
ax1.legend(loc = 2)

ax2.set_xlabel("Income")
ax2.set_ylabel("Distance from City Center")
ax2.scatter(X[log00, 0], X[log00, 1], c='#2c7a5d', edgecolors='#10553c', marker='o')
ax2.scatter(X[log01, 0], X[log01, 1], c='#CC6600', edgecolors='#95682D',
           marker='o')
ax2.scatter(X[log10, 0], X[log10, 1], c='#00cc00', edgecolors='#006600', marker='s')
ax2.scatter(X[log11, 0], X[log11, 1], c='#fc9f00', edgecolors='#E9890A',
           marker='s')
ax2.scatter(fair_model.w_[0, 0], fair_model.w_[0, 1], c='#2c7a5d', edgecolors='#10553c',
           linewidths=2, s=150, marker='D')
ax2.scatter(fair_model.w_[1, 0], fair_model.w_[1, 1], c='#CC6600', edgecolors='#95682D', linewidths=2, s=150, marker='D')
ax2.set_title('alpha = 200')
ax3.set_xlabel("Income")
ax3.set_ylabel("Distance from City Center")
ax3.scatter(X[log00, 0], X[log00, 1], c='#2c7a5d', edgecolors='#10553c', marker='o')
ax3.scatter(X[log01, 0], X[log01, 1], c='#CC6600', edgecolors='#95682D',
            marker='o')
ax3.scatter(X[log10, 0], X[log10, 1], c='#00cc00', edgecolors='#006600', marker='s')
ax3.scatter(X[log11, 0], X[log11, 1],  c='#fc9f00', edgecolors='#E9890A',
            marker='s')
ax3.scatter(norm_fair_model.w_[0, 0], norm_fair_model.w_[0, 1],  c='#2c7a5d', edgecolors='#10553c',
            linewidths=2, s=150, marker='D')
ax3.scatter(norm_fair_model.w_[1, 0], norm_fair_model.w_[1, 1], c='#CC6600', edgecolors='#95682D', linewidths=2, s=150, marker='D')
ax3.set_title('alpha = 400')
f.set_size_inches(18.5,10.5)
f.savefig('./evaluation/02ratio.eps', format='eps')
plt.show()