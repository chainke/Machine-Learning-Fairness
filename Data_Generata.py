import numpy as np
import math
import random


"""
To make distinction easier the first dimension will always represent the protected feature and its unprotected counter
part

The first try will create sample data for the credit problem.
We will choose gender as our fairness feature. 0 is female 1 is male.
We have to come up with a certain correlation between the protected feature  and the correlation to simulate the problem.



Example
v = [0]   protected feature
    []
    []
    []
    [ ]   classification
"""

def generate_protected_feature_data(number_of_points, proportion, seed=13):

    random.seed(seed)

    v = list(np.zeros(number_of_points))

    for i in range(0, number_of_points):

        if (random.random() >=proportion):
            v[i] = 1

    return v


def generate_children_feature(data):

    new_data = [[data[i], random.randint(1,5)] for i in range(0,len(data))]

    print(new_data)

    return new_data


n = 100
p = 0.2

v = generate_protected_feature_data(n, p)
v_c = generate_children_feature(v)

# print(v)
# sum = 0;
#
# for i in range(0,n):
#     sum += v[i]
#
# sum = sum/n
#
# print(sum)