import numpy as np
import matplotlib.pyplot as plt
import measures.functions as measure
import data.demo_benjamin as unfair_benjamin
import data.random_data_glvq as random_data
import mutual_information_Weights as mi

print(__doc__)


# use this to bring data generation and fairness measurement together as shown below:
# - import data generation (and if necessary classifier, or classify in your data generation class)
# - use the measures from zliobaite_measures to estimate your fairness
# should work if you run this script from repo root folder



#################################################
#
# unfair data from benjamin with
#
#################################################

print('\n\n#################################\n#pipeline with unfair data\n#################################\n')
print("GLVQ:\n")

unfairX, unfairY, unfairY_predicted = unfair_benjamin.getData()
protected = unfair_benjamin.getProtected()

model = unfair_benjamin.getTrainedModel()

print('classification accuracy:', model.score(unfairX, unfairY_predicted))

# process data, since unfairY_predicted contains boolean
unfairY_processed = []
unfairY_predicted_processed = []
for i in range(len(unfairY_predicted)):
	if(unfairY_predicted[i]):
		unfairY_predicted_processed.append(1)
	else:
		unfairY_predicted_processed.append(0)

	if(unfairY[i]):
		unfairY_processed.append(1)
	else:
		unfairY_processed.append(0)

# fairness measures from unfair data
measure.printAbsoluteMeasures(unfairY_predicted_processed, protected)


# grlvq with mutual information

print("\n\nGRLVQ:\n")

weights, predicted = mi.grlvq_fit(unfairX, unfairY_processed, protected)

measure.printAbsoluteMeasures(predicted.tolist(), protected)

#mi.run_mi_data_generata()


#################################################
#
# random data for maximal fairness and comparison
#
#################################################

nb_ppc = 100
#print('GLVQ:')


print('\n\n#################################\n#pipeline with random data\n#################################\n')

toy_data, pred = random_data.getData()
toy_protected = random_data.getProtected()
glvq = random_data.getTrainedModel()

print('classification accuracy:', glvq.score(toy_data, pred))

# fairness measures random for comparison
measure.printAbsoluteMeasures(pred.tolist(), toy_protected)
