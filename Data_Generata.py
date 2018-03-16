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



Example
v = []  protected feature (gender, 1 = female)
    []  number of children (between 0 and 5)
    []  annual income (between 0 and 50000)
    []  time unemployed (see above)
    []  married (1 for married)
"""


class CreditData:

    __max_children = 5
    __max_income = 50000
    __max_unemployed = 10
    __approval_income = 20000
    __approval_absent_time = 7
    __approval_married = 1
    __number_of_criteria = 3

    def __init__(self, child, income, unemployed, ap_income, ap_unemployed, number_criteria):
        self.__max_children = child
        self.__max_unemployed= unemployed
        self.__max_income = income
        self.__approval_income = ap_income
        self.__approval_absent_time = ap_unemployed
        self.__number_of_criteria = number_criteria
        self.__approval_married = 1

    def generate_protected_feature_data(self, number_of_points, proportion_of_males, seed=13):

        random.seed(seed)

        v = list(np.zeros(number_of_points))

        for i in range(0, number_of_points):
            if random.random() >= proportion_of_males:
                v[i] = 1

        return v

    def generate_other_features(self, data, gap_between_groups):

        new_data = [[data[i],                                   # gender
                     random.randint(0, self.__max_children),    # children
                     random.randint(0, self.__max_income),      # income
                     0,   # gets filled out further below       # time unemployed, depends on number of children
                     random.randint(0, 1)]                      # married
                    for i in range(0, len(data))]

        for i in range(0, len(data)):
            new_data[i][3] = random.uniform(0, self.__max_unemployed) + \
                             new_data[i][1] * (0.5 + 1*(math.exp(gap_between_groups)-1) *
                                               (new_data[i][0] - 0.5 * (1-new_data[i][0])))  # TODO: improve further

        return new_data

    def classify(self, data):

        y = list(np.zeros(len(data)))

        for i in range(0, len(data)):
            approval = 0
            if data[i][2] >= self.__approval_income:
                approval = approval+1
            if data[i][3] < self.__approval_absent_time:
                approval = approval+1
            if data[i][4] == self.__approval_married:
                approval = approval+1

            if approval >= self.__number_of_criteria-1:
                y[i] = 1

        return y

    def generate_credit_data(self, n, number_males, gap_between_groups, seed=13):
        X = self.generate_other_features(self.generate_protected_feature_data(n, number_males, seed),
                                         gap_between_groups)
        y = self.classify(X)
        return X, y

    def generate_credit_data_debug(self, n, number_males, gap_between_groups, seed=13):
        X, y = self.generate_credit_data(n, number_males, gap_between_groups, seed)
        men, women = self.print_data(X, y, number_males, gap_between_groups)
        return X, y, men, women

    def print_data(self, v_c, y, p, gap):

        n = len(v_c)

        print(n)

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
            approved += y[i]
            if y[i] == 1:
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

        # print("average gender:           " + str(gender))
        # print("average #children:        " + str(children))
        # print("average income:           " + str(income))
        # print("average time unemployed:  " + str(time))
        # print("average #people married:  " + str(married))
        # print("average #people approved: " + str(approved))
        print("average #men approved:    " + str(men_approved))
        print("average #women approved:  " + str(women_approved))
        print("gap in between:           " + str(men_approved-women_approved))
        print("gap put in:               " + str(gap))
        print("difference:               " + str(gap-men_approved+women_approved))

        return men_approved, women_approved
