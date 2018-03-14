import numpy as np
from zliobaite import generate_data
import Data_Generata


def divide_zliobaite_by_Group(data):
    male = []
    female = []
    for i in range(0, len(data)):
        if data[i][2] == 'M':
            male.append([data[i][0],data[i][1]])
        else:
            female.append([data[i][0],data[i][1]])
    return male, female


def divide_credit_data(data):
    male = []
    female = []
    for i in range(0, len(data)):
        if data[i][0] == 0:
            data[i].pop(0)
            male.append(data[i])
        else:
            data[i].pop(0)
            female.append(data[i])
    return male, female


test = generate_data(10, 0.2, 0.5, 0.5)
print(test)
testmale, testfemale = divide_zliobaite_by_Group(test)
print("male :")
print(testmale)
print("female :")
print(testfemale)


