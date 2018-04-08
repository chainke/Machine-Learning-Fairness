# Fair GLVQ Demo
#
# (c) 2018 AG ML
# CITEC Center of Excellence
# Bielefeld University

import math
import numpy as np
import matplotlib.pyplot as plt
from glvq.plot_2d import tango_color
from glvq.plot_2d import to_tango_colors
from glvq.plot_2d import plot2d
from glvq.glvq import GlvqModel
from sklearn.utils import shuffle

# DON'T USE AFTER GENERATOR CLASS IS WORKING
# TODO: use generator class for data generation. Currently TWO versions of the same thing

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
m     = 1000
# Gaussian standard deviation for distance from city center
std_1 = 0.2
# Gaussian standard deviation for income
std_2 = 0.2

# proportion of non-white people who do not pay their money back (if 0.5 non-discriminatory
# classification is trivial)
p0    = 0.8
# proportion of white people who do not pay their money back
p1    = 0.5
# proportion of non-white people in the overall data set
q     = 0.5

# proportion of non-white people in urban neighbourhoods
q_urban = 0.8
# proportion of non-white people in suburban neighbourhoods
q_suburban = 1 - q_urban

# proportion of white people in urban neighbourhoods
p_urban = 0.5

# number of people living in urban neighbourhoods
m_urban = int(m * (1 - q) * p_urban + m * q * q_urban)
# number of people living in suburban neighbourhoods
m_suburb = m - m_urban

print("----")
print("Total number of people in urban neighbourhoods: \t{}".format(m_urban))
print("Total number of people in suburban neighbourhoods: \t{}".format(m_suburb))
print("----")

X_urban = np.random.randn(m_urban, 2).dot(np.array([[std_1, 0], [0, std_2]]))
X_suburb = np.random.randn(m_suburb, 2).dot(np.array([[std_1, 0], [0, std_2]]))

# shift suburban population
X_suburb += np.array([1, 0])

# sort data points in both sets
X_urban_sorted = sorted(X_urban, key=lambda x: x[1], reverse=False)

# generate a vector C denoting the racial information
C = []
C.append(np.zeros(m_urban, dtype=bool))
# m0    = int(m_urban * q_urban * q)
m0 = int(q_urban * q * m)
# q * m = 0.5 * 1000 = 500                  non-whites
# q_urban * q * m = 0.8 * 0.5 * 1000 = 400
#
C[0][m0:] = True
print("Number of \tnon-white \turban: \t\t{}".format(m0))
print("Number of \twhite \t\turban: \t\t{}".format(m_urban- m0))

X_suburb_sorted = sorted(X_suburb, key=lambda x: x[1], reverse=False)
C.append(np.zeros(m_suburb, dtype=bool))
m1    = int(m * q_suburban * q)
C[1][m1:] = True
print("Number of \tnon-white \tsuburban: \t{}".format(m1))
print("Number of \twhite \t\tsuburban: \t{}".format(m_suburb - m1))

print("----")

# assign class in proportion of subgroup and location
# -> should sum up correctly

# generate a vector Y for urban area denoting the actual information about whether
# money was paid back or not
Y = []
Y.append(np.zeros(m_urban, dtype=bool))
# set
m00   = int(m0 * p0)
Y[0][m00:m0] = True
Y[0][:m0] = shuffle(Y[0][:m0])
print("Number of \turban \t\tnon-white \tdo not pay\t(m00): {}".format(m00))
m10   = int((m_urban - m0) * p1)
Y[0][m0+m10:] = True
Y[0][m0:] = shuffle(Y[0][m0:])
print("Number of \turban \t\twhite \t\tdo not pay\t(m10): {}".format(m10))

# generate a vector Y for suburban area denoting the actual information about whether
# money was paid back or not
Y.append(np.zeros(m_suburb, dtype=bool))
m00   = int(m1 * p0)
Y[1][m00:m0] = True
Y[1][:m0] = shuffle(Y[1][:m0])
print("Number of \tsuburban \tnon-white \tdo not pay\t(m00): {}".format(m00))
m10   = int((m_suburb - m1) * p1)
Y[1][m0+m10:] = True
Y[1][m0:] = shuffle(Y[1][m0:])
print("Number of \tsuburban \twhite \t\tdo not pay\t(m10): {}".format(m10))
print("----")

X_full = np.concatenate((X_urban_sorted, X_suburb_sorted))
C_full = np.concatenate((C[0], C[1]))
Y_full = np.concatenate((Y[0], Y[1]))


# non-white: do not pay
log00 = np.logical_and(np.logical_not(C_full), np.logical_not(Y_full))
# non-white: pay
log01 = np.logical_and(np.logical_not(C_full), Y_full)

# white: do not pay
log10 = np.logical_and(C_full, np.logical_not(Y_full))
# white: pay
log11 = np.logical_and(C_full, Y_full)

# Train a GLVQ model
model = GlvqModel()
model.fit(X_full, Y_full)

# Plot the data and the prototypes as well
fig = plt.figure()
fig.canvas.set_window_title("LVQ with continuous distance to city center")
ax  = fig.add_subplot(111)
ax.set_xlabel("Distance from City Center")
ax.set_ylabel("Income")

color_0 = 'skyblue'
color_1 = 'scarletred'
color_pay = 'butter'

# non-white: do not pay
ax.scatter(X_full[log00, 0], X_full[log00, 1], c=tango_color(color_0, 0),
           edgecolors=tango_color(color_0, 2), marker='o')
# non-white: pay
ax.scatter(X_full[log01, 0], X_full[log01, 1], c=tango_color(color_pay, 0),
           edgecolors=tango_color(color_0, 2), marker='o')
# white: do not pay
ax.scatter(X_full[log10, 0], X_full[log10, 1], c=tango_color(color_1, 0),
           edgecolors=tango_color(color_1, 2), marker='s')
# white: pay
ax.scatter(X_full[log11, 0], X_full[log11, 1], c=tango_color(color_pay, 0),
           edgecolors=tango_color(color_1, 2), marker='s')


# ax.scatter(X_full[log00, 0], X_full[log00, 1], c=tango_color(color_0, 0), edgecolors=tango_color(color_0, 2), marker='o')
# ax.scatter(X_full[log01, 0], X_full[log01, 1], c=tango_color(color_1, 0), edgecolors=tango_color(color_1, 2), marker='o')
# ax.scatter(X_full[log10, 0], X_full[log10, 1], c=tango_color(color_0, 0), edgecolors=tango_color(color_0, 2), marker='s')
# ax.scatter(X_full[log11, 0], X_full[log11, 1], c=tango_color(color_1, 0), edgecolors=tango_color(color_1, 2), marker='s')


ax.scatter(model.w_[0, 0], model.w_[0, 1], c=tango_color(color_0, 1), edgecolors=tango_color(color_0, 2), linewidths=2, s=150, marker='D')
ax.scatter(model.w_[1, 0], model.w_[1, 1], c=tango_color(color_1, 1), edgecolors=tango_color(color_1, 2), linewidths=2, s=150, marker='D')
plt.show()


# # Plot the data and the prototypes as well
# fig = plt.figure()
# fig.canvas.set_window_title("LVQ with continuous distance to city center")
# ax  = fig.add_subplot(111)
# ax.set_xlabel("Distance from City Center")
# ax.set_ylabel("Income")
#
# ax.scatter(X_urban[:, 0], X_urban[:, 1], c=tango_color('butter', 0), edgecolors=tango_color('butter', 2), marker='o')
# ax.scatter(X_suburb[:, 0], X_suburb[:, 1], c=tango_color(color_1, 0), edgecolors=tango_color(color_1, 2), marker='o')
#
#
# plt.show()
