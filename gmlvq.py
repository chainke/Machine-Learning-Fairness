import numpy as np
import Data_Generata
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split



n = 100000
p = 0.5
data = Data_Generata.CreditData(5,50000,10, 20000, 7,2).generate_credit_data(n,p)

X = []
y = []
for i in range (0,len(data)):
    y.append(data[i].pop())
    X.append(data[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/2)
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data)
distances, indices = nbrs.kneighbors(data)


sum = 0.0
positivelabel = 0
positivedata = 0

for i in range (0,len(data)):

    if indices[i] == data[i][4]:
        sum += 1

print("accuracy   :" + str(sum/(len(data))))

