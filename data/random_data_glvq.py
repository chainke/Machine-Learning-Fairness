import numpy as np
import matplotlib.pyplot as plt

from GLVQ.glvq import GlvqModel
from GLVQ.plot_2d import to_tango_colors, tango_color

# generate random data
toy_data = np.append(
    np.random.multivariate_normal([0, 0], np.eye(2) / 2, size=100),
    np.random.multivariate_normal([5, 0], np.eye(2) / 2, size=100), axis=0)
toy_label = np.append(np.zeros(100), np.ones(100), axis=0)



# model fitting
glvq = GlvqModel()
glvq.fit(toy_data, toy_label)
pred = glvq.predict(toy_data)
#print(pred)

# plotting
plt.scatter(toy_data[:, 0], toy_data[:, 1], c=to_tango_colors(toy_label), alpha=0.5)
plt.scatter(toy_data[:, 0], toy_data[:, 1], c=to_tango_colors(pred), marker='.')
plt.scatter(glvq.w_[:, 0], glvq.w_[:, 1],
            c=tango_color('aluminium', 5), marker='D')
plt.scatter(glvq.w_[:, 0], glvq.w_[:, 1],
            c=to_tango_colors(glvq.c_w_, 0), marker='.')
plt.axis('equal')

#plt.show()


def getData():
	return toy_data, toy_label

def getProtected():
	# generate list of protected group: 
	# for every label, the half of it belongs to the protected group
	# should produce maximal fairness
	toy_protected = []
	for i in range(int(len(toy_label)/4)):
		toy_protected.append(0)

	for i in range(int(len(toy_label)/4)):
		toy_protected.append(1)

	for i in range(int(len(toy_label)/4)):
		toy_protected.append(0)

	for i in range(int(len(toy_label)/4)):
		toy_protected.append(1)

	return toy_protected

def getTrainedModel():
	return glvq
