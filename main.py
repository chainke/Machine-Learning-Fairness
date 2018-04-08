import numpy as np
import matplotlib.pyplot as plt
import measures.functions as measure
import data.demo_benjamin as unfair_benjamin
import data.random_data_glvq as random_data
import mutual_information_Weights as mi
import Data_Generata
import sys
from GLVQ.glvq import GlvqModel

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

def data_benjamin():

	print('\n\n#################################\n#pipeline with unfair data\n#################################\n')
	print("GLVQ:\n")

	# data from benjamin
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

#################################################
#
# unfair data from credit example
#
#################################################

def data_credit():
	print('\n\n#################################\n#pipeline with credit data\n#################################\n')

	n = 10000#0
	p = 0.9
	X, y = Data_Generata.CreditData(5,50000,10, 20000, 7,2).generate_credit_data(n,p,0.5)
	
	protected = []
	for i in range(0, len(X)):
		protected.append(X[i][0])

	print('GLVQ:\n')

	glvq = GlvqModel()
	glvq.fit(X,y)
	predicted_glvq = glvq.predict(X)

	print('classification accuracy:', glvq.score(X, predicted_glvq))
	measure.printAbsoluteMeasures(predicted_glvq.tolist(), protected)

	
	print('\n\nGLRVQ:\n')

	weights, predicted_glrvq = mi.grlvq_fit(X, y, protected)

	measure.printAbsoluteMeasures(predicted_glrvq.tolist(), protected)

	print('\n\nGLVQ with weighted preprocessing:\n')

	newX = mi.weighted_preprocessing(X, y, protected)

	glvq = GlvqModel()
	glvq.fit(newX,y)
	predicted_glvq = glvq.predict(newX)

	print('classification accuracy:', glvq.score(newX, predicted_glvq))
	measure.printAbsoluteMeasures(predicted_glvq.tolist(), protected)



#################################################
#
# random data for maximal fairness and comparison
#
#################################################

def data_random():
	nb_ppc = 100
	
	print('\n\n#################################\n#pipeline with random data\n#################################\n')
	print('GLVQ:')

	toy_data, pred = random_data.getData()
	toy_protected = random_data.getProtected()
	glvq = random_data.getTrainedModel()

	print('classification accuracy:', glvq.score(toy_data, pred))

	# fairness measures random for comparison
	measure.printAbsoluteMeasures(pred.tolist(), toy_protected)


if(len(sys.argv) != 2):
	print("please pass the argument 'benjamin' for data from benjamin or 'credit' for credit example")
	sys.exit()

if (sys.argv[1] == "benjamin"): # use data from benjamin
	data_benjamin()
elif (sys.argv[1] == "credit"): # use credit example
	data_credit()

data_random()
		