import numpy as np
import matplotlib.pyplot as plt
import measures.functions as measure
import fairness_demo
import random_data_glvq

print(__doc__)


# use this to bring data generation and fairness measurement together as shown below:
# - import data generation (and if necessary classifier, or classify in your data generation class)
# - use the measures from zliobaite_measures to estimate your fairness
# should work if you run this script from repo root folder



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

# fairness measures from unfair data
measure.printAbsoluteMeasures(unfairY_predicted_processed, protected)



#################################################
#
# random data for maximal fairness and comparison
#
#################################################

nb_ppc = 100
#print('GLVQ:')


print('\n\n#################################\n#pipeline with random data\n#################################\n')

toy_data, pred = random_data_glvq.getData()
toy_protected = random_data_glvq.getProtected()
glvq = random_data_glvq.getTrainedModel()

print('classification accuracy:', glvq.score(toy_data, pred))

# fairness measures random for comparison
measure.printAbsoluteMeasures(pred.tolist(), toy_protected)