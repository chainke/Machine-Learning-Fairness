import numpy as np
import Data_Generata as generate
from GlvqUtility import GlvqUtility
from Platt_Scaling_LVQ.glvq import GlvqModel
from sklearn.model_selection import train_test_split

n = 12000
p = 0.0
intended_gap = 0.5

X, y = generate.CreditData(5, 50000, 10, 20000, 7, 2).generate_credit_data(n, p, intended_gap)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# model fitting
glvq = GlvqModel()
glvq_utility = GlvqUtility(glvq)
glvq.fit(X_train, y_train)
prototypes = glvq.w_
pred = glvq_utility.get_prediction_accuracy(prototypes, X_test, y_test)
pred_capped = 1 / (1 + np.exp(-pred))
print(pred_capped)

# predict = glvq.predict(X_test)
group = [X_test[i][0] for i in range(len(X_test))]
label = y_test

savedata = [['', 'label', 'group', 'prediction']]

for i in range(len(pred_capped)):
    savedata.append([i, label[i], group[i], pred_capped[i]])

    print(savedata[i])

np.savetxt('relaxed_equalized_odds/data/data.csv', savedata, delimiter=',', fmt='%s')