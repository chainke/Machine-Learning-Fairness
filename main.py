import numpy as np
import matplotlib.pyplot as plt
import measures.zliobaite_measures as measure
import fairness_demo

from GLVQ.glvq import GlvqModel
from GLVQ.plot_2d import to_tango_colors, tango_color

print(__doc__)

# TODO: split in several files or functions for easy swap of e.g. classifier

nb_ppc = 100
print('GLVQ:')

#################################################
#
# random data for maximal fairness and comparison
#
#################################################

# generate random data
toy_data = np.append(
    np.random.multivariate_normal([0, 0], np.eye(2) / 2, size=100),
    np.random.multivariate_normal([5, 0], np.eye(2) / 2, size=100), axis=0)
toy_label = np.append(np.zeros(100), np.ones(100), axis=0)
toy_protected = np.append(np.zeros(100), np.ones(100), axis=0)
#print(toy_label)
#print(toy_protected)

print('pipeline with random data for comparison')

# model fitting
glvq = GlvqModel()
glvq.fit(toy_data, toy_label)
pred = glvq.predict(toy_data)
#print(pred)
print('classification accuracy:', glvq.score(toy_data, toy_label))

# fairness measures random for comparison
fairness = measure.elift(pred.tolist(), toy_protected.tolist())
print('elift ratio: ', fairness)

fairness = measure.odds_ratio(pred.tolist(), toy_protected.tolist())
print('odds ratio: ', fairness)

fairness = measure.impact_ratio(pred.tolist(), toy_protected.tolist())
print('impact ratio: ', fairness)

fairness = measure.mean_difference(pred.tolist(), toy_protected.tolist())
print('mean difference: ', fairness)

fairness = measure.normalized_difference(pred.tolist(), toy_protected.tolist())
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



#################################################
#
# unfair data from benjamin
#
#################################################

print('pipeline with unfair data')

unfairX, unfairY_predicted = fairness_demo.getData()

# process data, since unfairY_predicted contains boolean
unfairY_predicted_processed = []
for i in range(len(unfairY_predicted)):
	if(unfairY_predicted[i]):
		unfairY_predicted_processed.append(1)
	else:
		unfairY_predicted_processed.append(0)

# fairness measures random for comparison
fairness = measure.elift(unfairY_predicted_processed, toy_protected.tolist())
print('elift ratio: ', fairness)

fairness = measure.odds_ratio(unfairY_predicted_processed, toy_protected.tolist())
print('odds ratio: ', fairness)

fairness = measure.impact_ratio(unfairY_predicted_processed, toy_protected.tolist())
print('impact ratio: ', fairness)

fairness = measure.mean_difference(unfairY_predicted_processed, toy_protected.tolist())
print('mean difference: ', fairness)

fairness = measure.normalized_difference(unfairY_predicted_processed, toy_protected.tolist())
print('normalized difference: ', fairness)