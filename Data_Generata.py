import numpy as np
import math
import random


"""
To make distinction easier the first dimension will always represent the protected feature and its unprotected counter
part

The first try will create sample data for the credit problem.
We will choose gender as our fairness feature. 0 is female 1 is male.
We have to come up with a certain correlation between the protected feature  and the correlation to simulate the problem.

We will distinct between 5 features.
Gender {0,1}
Annual income [0,..,50.000]
Children {0,..,5}
Time unemployed in years  {0,...,10} (We assume a typical person starts works 40 years and will spent maximal 10 percent unemployed if they apply for a credit.)
Married {0,1}

The indirect discrimination will be settled as follows:

The time unemployed will be computed like this:

female: r + 0.5K + 0.5K = r + K
male: r + 0.5K

where K = number of children
      r = random value for time unemployed, between 0 and 10

The first 0.5 assumes that each parent will take 6 months parental vacation, but females wont be able to work the second
half of their pregnancy, which we assume as additional 6 months.

A person will receive a credit (y=1) if they fulfill 2 of the 3 criteria:

        Income >= 20.000
        Unemployed < 7 (around average)
        Married = 1


Therefore a woman with 3 children and 0 years unemployed will be classified as 0, a male will never be classified as 0 depending on the
amount of children with perfect work attendance.


Example
v = []  protected feature (gender, 1 = female)
    []  number of children (between 0 and 5)
    []  annual income (between 0 and 50000)
    []  time unemployed (see above)
    []  married (1 for married)
"""

max_children = 5
max_income = 50000
max_unemployed = 10

approval_income = 20000
approval_absent_time = 7
approval_married = 1

number_of_criteria = 2


def generate_protected_feature_data(number_of_points, proportion_of_males, seed=13):

    random.seed(seed)

    v = list(np.zeros(number_of_points))

    for i in range(0, number_of_points):
        if random.random() >= proportion_of_males:
            v[i] = 1

    return v


def generate_other_features(data):

    new_data = [[data[i],                           # gender
                 random.randint(0, max_children),   # children
                 random.randint(0, max_income),     # income
                 0,                                 # time unemployed, depends on number of children
                 random.randint(0, 1)]              # married
                for i in range(0, len(data))]

    for i in range(0, len(data)):
        new_data[i][3] = random.uniform(0, max_unemployed) + new_data[i][1] * (0.5 + 0.5*new_data[i][0])

    return new_data


def classify(data):

    for i in range(0, len(data)):
        data[i].append(0)
        approval = 0
        if data[i][2] >= approval_income:
            approval = approval+1
        if data[i][3] < approval_absent_time:
            approval = approval+1
        if data[i][4] == approval_married:
            approval = approval+1

        if approval >= number_of_criteria:
            data[i][5] = 1


n = 10000
p = 0.5

v = generate_protected_feature_data(n, p)
v_c = generate_other_features(v)
classify(v_c)

print(v_c)
gender = 0
children = 0
income = 0
time = 0
married = 0
approved = 0
men_approved = 0
women_approved = 0

for i in range(0,n):
    gender += v_c[i][0]
    children += v_c[i][1]
    income += v_c[i][2]
    time += v_c[i][3]
    married += v_c[i][4]
    approved += v_c[i][5]
    if v_c[i][5] == 1:
        if v_c[i][0] == 1:
            women_approved = women_approved+1
        if v_c[i][0] == 0:
            men_approved = men_approved+1

gender = gender/n
children = children/n
income = income/n
time = time/n
married = married/n
approved = approved/n
men_approved = men_approved/(n*p)
women_approved = women_approved/(n*(1-p))

print("average gender:           " + str(gender))
print("average #children:        " + str(children))
print("average income:           " + str(income))
print("average time unemployed:  " + str(time))
print("average #people married:  " + str(married))
print("average #people approved: " + str(approved))
print("average #men approved:    " + str(men_approved))
print("average #women approved:  " + str(women_approved))
