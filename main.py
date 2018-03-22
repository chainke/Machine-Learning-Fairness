import numpy as np
import matplotlib.pyplot as plt
import measures.zliobaite_measures as measure
import fairness_demo

from GLVQ.glvq import GlvqModel
from GLVQ.plot_2d import to_tango_colors, tango_color

print(__doc__)

# TODO: split in several files or functions for easy swap of e.g. classifier



#################################################
#
# unfair data from benjamin
#
#################################################

print('\n\n#################################\n#pipeline with unfair data\n#################################\n')

unfairX, unfairY_predicted = fairness_demo.getData()
protected = fairness_demo.getProtected()

model = fairness_demo.getTrainedModel()

print('classification accuracy:', model.score(unfairX, unfairY_predicted))

# process data, since unfairY_predicted contains boolean
unfairY_predicted_processed = []
for i in range(len(unfairY_predicted)):
	if(unfairY_predicted[i]):
		unfairY_predicted_processed.append(1)
	else:
		unfairY_predicted_processed.append(0)

# fairness measures random for comparison
fairness = measure.elift(unfairY_predicted_processed, protected)
print('elift ratio: ', fairness)

fairness = measure.odds_ratio(unfairY_predicted_processed, protected)
print('odds ratio: ', fairness)

fairness = measure.impact_ratio(unfairY_predicted_processed, protected)
print('impact ratio: ', fairness)

fairness = measure.mean_difference(unfairY_predicted_processed, protected)
print('mean difference: ', fairness)

fairness = measure.normalized_difference(unfairY_predicted_processed, protected)
print('normalized difference: ', fairness)



#################################################
#
# random data for maximal fairness and comparison
#
#################################################

nb_ppc = 100
#print('GLVQ:')

# generate random data
toy_data = np.append(
    np.random.multivariate_normal([0, 0], np.eye(2) / 2, size=100),
    np.random.multivariate_normal([5, 0], np.eye(2) / 2, size=100), axis=0)
toy_label = np.append(np.zeros(100), np.ones(100), axis=0)

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

#print(toy_label)
#print(toy_protected)

print('\n\n#################################\n#pipeline with random data\n#################################\n')

# model fitting
glvq = GlvqModel()
glvq.fit(toy_data, toy_label)
pred = glvq.predict(toy_data)
#print(pred)
print('classification accuracy:', glvq.score(toy_data, pred))

# fairness measures random for comparison
fairness = measure.elift(pred.tolist(), toy_protected)
print('elift ratio: ', fairness)

fairness = measure.odds_ratio(pred.tolist(), toy_protected)
print('odds ratio: ', fairness)

fairness = measure.impact_ratio(pred.tolist(), toy_protected)
print('impact ratio: ', fairness)

fairness = measure.mean_difference(pred.tolist(), toy_protected)
print('mean difference: ', fairness)

fairness = measure.normalized_difference(pred.tolist(), toy_protected)
print('normalized difference: ', fairness)


# plotting
plt.scatter(toy_data[:, 0], toy_data[:, 1], c=to_tango_colors(toy_label), alpha=0.5)
plt.scatter(toy_data[:, 0], toy_data[:, 1], c=to_tango_colors(pred), marker='.')
plt.scatter(glvq.w_[:, 0], glvq.w_[:, 1],
            c=tango_color('aluminium', 5), marker='D')
plt.scatter(glvq.w_[:, 0], glvq.w_[:, 1],
            c=to_tango_colors(glvq.c_w_, 0), marker='.')
plt.axis('equal')

#plt.show()

