# Fair GLVQ Demo
#
# (c) 2018 AG ML
# CITEC Center of Excellence
# Bielefeld University

# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn_lvq.utils import _tango_color
# from GLVQ.plot_2d import tango_color
# from sklearn_lvq.glvq.plot_2d import to_tango_colors
# from sklearn_lvq.glvq.plot_2d import plot2d
# from sklearn.utils import shuffle

from sklearn_lvq.glvq import GlvqModel
import data.generator as generator


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

gen = generator.DataGen()
X, C, Y = gen.generate_two_bubbles(number_data_points=m, proportion_0=q, proportion_0_urban=q_urban,
                                   proportion_1_urban=p_urban, proportion_0_pay=1-p0, proportion_1_pay=p1)

# Train a GLVQ model
model = GlvqModel()
model.fit(X, Y)
Y_pred = model.predict(X)
print("Y_pred: {}".format(Y_pred))

ax = gen.prepare_plot(X=X, C=C, Y=Y, Y_pred=Y_pred, prototypes=model.w_)

gen.plot_prepared_dist(ax)

