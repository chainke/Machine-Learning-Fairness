import numpy as np
from GlvqUtility import GlvqUtility
from sklearn.model_selection import train_test_split
from data.generator import DataGen
from sklearn_lvq.glvq import GlvqModel
from fair_glvq import MeanDiffGlvqModel as FairGlvqModel

n = 12000
proportion_0 = 0.5
proportion_0_urban = 0.5
proportion_1_urban = 0.5
proportion_0_pay = 0.5
proportion_1_pay = 0.5

generator = DataGen()

# X, y = generate.CreditData(5, 50000, 10, 20000, 7, 2).generate_credit_data(n, p, intended_gap)
x, c, y = generator.generate_two_bubbles(n, proportion_0, proportion_0_urban, proportion_1_urban,
                                         proportion_0_pay, proportion_1_pay)

X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(x, y, (1-c), test_size=1/3, random_state=42)

# model fitting
# fair_glvq = FairGlvqModel()
# glvq_utility = GlvqUtility(fair_glvq)
# fair_glvq.fit_fair(X_train, y_train, c_train)
# prototypes = fair_glvq.w_
# pred = glvq_utility.get_prediction_accuracy(prototypes, X_test, y_test)
# pred_capped = 1 / (1 + np.exp(-pred))

glvq = GlvqModel()
glvq_utility = GlvqUtility(glvq)
glvq.fit(X_train, y_train)
prototypes = glvq.w_
pred = glvq_utility.get_prediction_accuracy(prototypes, X_test, y_test)
pred_capped = 1 / (1 + np.exp(-pred))

# predict = glvq.predict(X_test)
group = c_test
label = y_test

savedata = [['', 'label', 'group', 'prediction']]

for i in range(len(pred_capped)):
    savedata.append([i, label[i], group[i], pred_capped[i]])

    print(savedata[i])

np.savetxt('relaxed_equalized_odds/data/data.csv', savedata, delimiter=',', fmt='%s')