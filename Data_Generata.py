# First idea for possible binary n dimensional feature vectors.
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
Time unemployed in years  {0,...,4} (We assume a typical person starts works 40 years and will spent maximal 10 percent unemployed if they apply for a credit.)
Married {0,1}

The indirect discrimination will be settled as follows:

The time unemployed will be computed like this:

female: r + 0.5K + 0.5K = r + K
male: r + 0.5K

where K = number of children
      r = random value for time unemployed

The first 0.5 assumes that each parent will take 6 months parental vacation, but females wont be able to work the second
half of their pregnancy, which we assume as additional 6 months.

A person will receive a credit (y=1) if they fulfill two of the 3 criteria:

        Income >= 20.000
        Unemployed < 3
        Married = 1


Therefore a woman with 3 children and 0 years unemployed will be classified as 0, a male will never be classified as 0 depending on the
amount of children with perfect work attendance.

"""