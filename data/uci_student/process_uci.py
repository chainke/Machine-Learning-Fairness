import csv
import numpy as np
import data.generator as generator

def process_uci_data():
	processed_data_name = "data/uci-student/dataset_processed.csv"
	feature_types = ["binary", "binary", "metric", "binary", "binary", 
	"binary", "category", "category", "category", "category", 
	"category", "category", "category", "category", "category", 
	"binary", "binary", "binary", "binary",	"binary", 
	"binary", "binary", "binary", "metric", "metric", 
	"metric", "metric", "metric", "metric", "metric", 
	"metric", "metric", "metric"]

	# read data
	csv_data = []

	with open("data/uci_student/student-por.csv", newline='') as csvfile:
	    data_reader = csv.reader(csvfile, delimiter=';', quotechar='|')

	    for row in data_reader:
	    	# only use the first 30 features because the last 3 are related to specific courses
	    	# if you want to use all data, you have to change the data of the last 3 features to floats
	        csv_data.append(row[:30])

	csv_data = np.array(csv_data[1:])
	print(csv_data[0:3])

	# normalize data
	num_rows, num_cols = csv_data.shape

	processed_data = np.zeros(num_rows)[np.newaxis]
	#print("initial preprocessed_data: {}".format(processed_data))
	for i in range(num_cols):
	    col = csv_data[:, i][np.newaxis]
	    #print(col)
	    #print("type of col: {}".format(type(col)))

	    if feature_types[i] is "binary":
	        processed_col = generator.normalize_binary_feature(col)
	    elif feature_types[i] is "unnormalized":
	        processed_col = generator.normalize_feature(col)
	    elif feature_types[i] is "categories":
	        processed_col = generator.normalize_category_feature(col)
	    elif feature_types[i] is "metric":
	        processed_col = generator.normalize_metric_feature(col)
	    elif feature_types[i] is "skip":
	        continue

	    #print("current processed_col: {}".format(processed_col))

	    #if verbose:
	        #print("i: {}\ttype: {} \tprocessed_data: {} \t processed_col: {}".format(i, feature_types[i],
	         #                                                                        processed_data.shape,
	          #                                                                       processed_col.shape))
	    processed_data = np.concatenate((processed_data, processed_col), axis=0)
	    #print("preprocessed_data of {}: {}".format(i, processed_data))

	processed_data = processed_data.T
	processed_data = processed_data[:,1:]
	print(processed_data[0:3])
	with open("data/uci_student/dataset_processed.csv", 'w', newline = '') as csvfile:
		dataWriter = csv.writer(csvfile, delimiter=';')
		for i in range(len(processed_data)):
			dataWriter.writerow(processed_data[i])

def get_students_data():

	# read data
	processed_data = []

	with open("data/uci_student/dataset_processed.csv", newline='') as csvfile:
	    data_reader = csv.reader(csvfile, delimiter=';', quotechar='|')

	    for row in data_reader:
	        processed_data.append(row)

	processed_data = np.array(processed_data)
	#print(processed_data[0:3])

	# make X, y, protected for fairness measurement and training
	X = []
	y = []
	protected = []

	# for using other labels and protected features, just change this numbers
	# attention: since the data is already processed, you have to find the 
	# collumns for label and protected in the processed_data.csv
	y_position = 15 # if it is school support
	protected_position = 1 # if it is gender

	for i in range(len(processed_data)):
		X_row = []

		for j in range(len(processed_data[i])):
			if(j == y_position):
				y.append(float(processed_data[i][j]))
			elif(j == protected_position):
				protected.append(float(processed_data[i][j]))
			else:
				X_row.append(float(processed_data[i][j]))
		X.append(X_row)

	#print(X[0:3])
	#print(y[0:3])
	#print(protected[0:3])

	return X, y, protected