import numpy as np
from zliobaite import generate_data


def divide_zliobaite_by_Group(data):
    male = []
    female = []
    for i in range(0, len(data)):
        if data[i][2] == 'M':
            male.append([data[i][0],data[i][1]])
        else:
            female.append([data[i][0],data[i][1]])
    return male, female



test = generate_data(10, 0.2, 0.5, 0.5)
print(test)
male, female = divide_zliobaite_by_Group(test)
print("male :")
print(male)
print("female :")
print(female)


