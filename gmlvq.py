import numpy as np
import Data_Generata
# import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split

training_list = []
validation_list = []

n = 500000
p = 0.5
k = 10

weights = 'uniform'

X, y = Data_Generata.CreditData(5,50000,10, 20000, 7,2).generate_credit_data(n,p)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)
X_train_true, X_valid, y_train_true, y_valid = train_test_split(X_train, y_train, test_size=1/2, random_state=42)

clf_training = neighbors.KNeighborsClassifier(k, weights=weights)
clf_training.fit(X_train_true, y_train_true)

training_score = clf_training.score(X_train_true, y_train_true)
validation_score = clf_training.score(X_valid, y_valid)

training_list.append(training_score)
validation_list.append(validation_score)

print("training score : {score} with k = {k}".format(k=k, score=training_score))
print("validation score : {score} with k = {k}".format(k=k, score=validation_score))

clf_test = neighbors.KNeighborsClassifier(k, weights=weights)
clf_test.fit(X_train_true, y_train_true)

best_k_test_score = clf_test.score(X_test, y_test)

print("test score : {score} with k = {k}".format(k=k, score=best_k_test_score))